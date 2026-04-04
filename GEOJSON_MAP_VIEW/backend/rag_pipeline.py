from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from openai import OpenAI

from backend.query_planner import QueryPlanner
from backend.spatial_engine import SpatialEngine
from backend.sql_engine import SQLEngine
from backend.vector_engine import VectorEngine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "gnem.duckdb"
DEFAULT_GEOJSON_PATH = PROJECT_ROOT / "data" / "Counties_Georgia.geojson"
DEFAULT_FAISS_PATH = PROJECT_ROOT / "data" / "gnem_faiss.index"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "vector_metadata.json"
QUESTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "capital",
    "does",
    "explain",
    "for",
    "from",
    "give",
    "how",
    "in",
    "is",
    "it",
    "list",
    "me",
    "of",
    "please",
    "show",
    "tell",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}


class HybridGeospatialRAGPipeline:
    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        geojson_path: Path = DEFAULT_GEOJSON_PATH,
        faiss_path: Path = DEFAULT_FAISS_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
    ) -> None:
        self.sql_engine = SQLEngine(db_path=db_path)
        self.spatial_engine = SpatialEngine(db_path=db_path, geojson_path=geojson_path)
        self.vector_engine = VectorEngine(faiss_path=faiss_path, metadata_path=metadata_path)
        self.query_planner = QueryPlanner(
            company_names=self.sql_engine.list_company_names(),
            county_names=self.spatial_engine.county_names,
        )

        self.llm_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
        self.llm_client = OpenAI(
            base_url=self.llm_base_url,
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            timeout=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60")),
            max_retries=0,
        )
        self.available_models = self._list_available_models()
        self.llm_model = os.getenv("OLLAMA_MODEL") or self._choose_default_model() or "retrieval-fallback"
        self.fallback_model_preferences = [
            "gemma3:27b",
            "qwen2.5:14b",
            "qwen2.5:7b",
            "llama3.1:8b",
            "mistral-small3.2:24b",
            "gemma3:12b",
            "qwen3:8b",
            "qwen3:4b",
            "qwen3:1.7b",
            "llama3.2:3b",
            "llama3.2:1b",
            "gemma3:4b",
            "tinyllama:latest",
        ]

    def answer_question(self, question: str) -> dict:
        plan = self.query_planner.plan(question)
        hints = plan.get("hints", {})
        analysis_intent = str(hints.get("analysis_intent") or "standard_retrieval")

        sql_df = pd.DataFrame()
        vector_df = pd.DataFrame()
        geo_df = pd.DataFrame()
        gap_report: Dict[str, object] = {}
        map_context: Dict[str, object] = self._build_initial_map_context(plan)

        if plan.get("sql"):
            sql_df = self._run_sql_retrieval(question, hints)

        if plan.get("vector"):
            top_k = 24 if analysis_intent in {"gap_analysis", "disruption_alternatives"} else (16 if plan.get("geo") else 10)
            vector_df = self.vector_engine.semantic_company_search(question, top_k=top_k, per_company_limit=6)
            vector_df = self._optional_keyword_filter(vector_df, question)
            vector_df = self._apply_structured_filters(vector_df, hints)
            if hints.get("oem") and analysis_intent != "disruption_alternatives":
                vector_df = self._filter_by_oem(vector_df, str(hints["oem"]))

        if plan.get("geo"):
            radius_km = float(hints.get("radius_km", 100.0))
            candidates = self._merge_candidates([vector_df, sql_df])
            candidate_input = None if candidates.empty else candidates

            coords = hints.get("coordinates")
            city = hints.get("city")
            counties = hints.get("counties") if isinstance(hints.get("counties"), list) else []
            company_name = str(hints.get("company_name") or "").strip()

            if analysis_intent == "disruption_alternatives" and company_name:
                geo_df = self.spatial_engine.rank_alternative_suppliers(
                    company_name=company_name,
                    radius_km=radius_km,
                    category_term=str(hints["category_term"]) if hints.get("category_term") else None,
                    capability_term=str(hints["capability_term"]) if hints.get("capability_term") else None,
                    max_results=25,
                )
                center_lat = geo_df["query_center_lat"].dropna().iloc[0] if not geo_df.empty and "query_center_lat" in geo_df.columns and geo_df["query_center_lat"].notna().any() else None
                center_lon = geo_df["query_center_lon"].dropna().iloc[0] if not geo_df.empty and "query_center_lon" in geo_df.columns and geo_df["query_center_lon"].notna().any() else None
                map_context.update(
                    {
                        "map_mode": "disruption_alternatives",
                        "focus_label": company_name,
                        "center_lat": float(center_lat) if center_lat is not None else None,
                        "center_lon": float(center_lon) if center_lon is not None else None,
                        "radius_km": radius_km,
                    }
                )
            elif counties and not city and not coords:
                geo_df = self.spatial_engine.companies_in_counties(
                    counties=counties,
                    candidates=candidate_input,
                )
                if geo_df.empty and candidate_input is not None:
                    geo_df = self.spatial_engine.companies_in_counties(counties=counties, candidates=None)
                map_context.update({"map_mode": "county_filter", "counties": counties})
            elif isinstance(coords, dict) and "lat" in coords and "lon" in coords:
                geo_df = self.spatial_engine.companies_within_radius(
                    lat=float(coords["lat"]),
                    lon=float(coords["lon"]),
                    radius_km=radius_km,
                    candidates=candidate_input,
                )
                map_context.update(
                    {
                        "map_mode": "radius_search",
                        "center_lat": float(coords["lat"]),
                        "center_lon": float(coords["lon"]),
                        "radius_km": radius_km,
                        "focus_label": f"{coords['lat']}, {coords['lon']}",
                    }
                )
            elif city:
                geo_df = self.spatial_engine.companies_near_city(
                    city_name=str(city),
                    radius_km=radius_km,
                    candidates=candidate_input,
                )
                resolved = self.spatial_engine.resolve_place_coordinates(str(city))
                map_context.update(
                    {
                        "map_mode": "radius_search",
                        "focus_label": str(city),
                        "center_lat": float(resolved[0]) if resolved else None,
                        "center_lon": float(resolved[1]) if resolved else None,
                        "radius_km": radius_km,
                    }
                )

            if geo_df.empty and candidate_input is not None:
                if isinstance(coords, dict) and "lat" in coords and "lon" in coords:
                    geo_df = self.spatial_engine.companies_within_radius(
                        lat=float(coords["lat"]),
                        lon=float(coords["lon"]),
                        radius_km=radius_km,
                        candidates=None,
                    )
                elif city:
                    geo_df = self.spatial_engine.companies_near_city(
                        city_name=str(city),
                        radius_km=radius_km,
                        candidates=None,
                    )

            geo_df = self._optional_keyword_filter(geo_df, question)
            geo_df = self._apply_structured_filters(geo_df, hints)
            if hints.get("oem") and analysis_intent != "disruption_alternatives":
                geo_df = self._filter_by_oem(geo_df, str(hints["oem"]))

        if analysis_intent in {"gap_analysis", "disruption_alternatives"}:
            gap_report = self.spatial_engine.supply_gap_report(
                capability_term=str(hints["capability_term"]) if hints.get("capability_term") else None,
                category_term=str(hints["category_term"]) if hints.get("category_term") else None,
                county_scope=hints.get("counties") if isinstance(hints.get("counties"), list) else None,
                max_gap_counties=12,
            )
            map_context["gap_report"] = gap_report

        if plan.get("geo"):
            final_df = geo_df.copy()
        else:
            final_df = self._choose_final_results(sql_df=sql_df, vector_df=vector_df, geo_df=geo_df)

        if final_df.empty and analysis_intent == "gap_analysis":
            final_df = self._merge_candidates([sql_df, vector_df])

        if self._should_reject_as_unsupported(
            question=question,
            plan=plan,
            sql_df=sql_df,
            vector_df=vector_df,
            geo_df=geo_df,
            final_df=final_df,
        ):
            return self._unsupported_response(question=question, plan=plan)

        final_df = self._annotate_map_weights(final_df, question=question, plan=plan)
        final_df = self._annotate_peer_distances(final_df)

        if plan.get("geo") and final_df.empty and analysis_intent != "gap_analysis":
            retrieved_chunks = [self._build_geo_no_results_chunk(question=question, hints=hints)]
        else:
            retrieved_chunks = self._build_retrieved_chunks(
                question=question,
                vector_df=vector_df,
                sql_df=sql_df,
                geo_df=geo_df,
                final_df=final_df,
                gap_report=gap_report,
                plan=plan,
            )
        map_context["result_count"] = int(len(final_df))
        map_context["county_coverage_count"] = int(final_df["county"].dropna().nunique()) if not final_df.empty and "county" in final_df.columns else 0
        context = self._format_context(
            question=question,
            plan=plan,
            retrieved_chunks=retrieved_chunks,
            final_df=final_df,
            map_context=map_context,
        )
        answer = self._generate_answer_with_llm(
            question=question,
            context=context,
            retrieved_chunks=retrieved_chunks,
            preferred_companies=final_df["company"].dropna().astype(str).tolist() if "company" in final_df.columns else [],
        )

        return {
            "answer": answer,
            "sources": [self._chunk_source_line(chunk) for chunk in retrieved_chunks[:12]],
            "retrieved_chunks": retrieved_chunks,
            "retrieved_companies": self._df_to_records(final_df.head(25)),
            "plan": plan,
            "map_context": map_context,
            "model_used": self.llm_model,
        }

    def _choose_default_model(self) -> Optional[str]:
        preferred = [
            "gemma3:27b",
            "gpt-oss:20b",
            "qwen2.5:14b",
            "qwen2.5:7b",
            "llama3.1:8b",
            "mistral-small3.2:24b",
            "gemma3:12b",
            "qwen3:14b",
            "qwen3:8b",
            "deepseek-r1:14b",
            "deepseek-r1:8b",
            "llama3.3:70b",
            "llama3.3:8b",
            "mistral-small3.1:24b",
            "gemma3:4b",
            "llama3.2:3b",
            "tinyllama:latest",
        ]
        model_ids = self.available_models
        if not model_ids:
            return None
        for candidate in preferred:
            if candidate in model_ids:
                return candidate
        return model_ids[0]

    def _list_available_models(self) -> List[str]:
        try:
            models = self.llm_client.models.list()
            return [m.id for m in getattr(models, "data", []) if getattr(m, "id", None)]
        except Exception:
            return []

    @staticmethod
    def _build_initial_map_context(plan: Dict[str, object]) -> Dict[str, object]:
        hints = plan.get("hints", {}) if isinstance(plan, dict) else {}
        coords = hints.get("coordinates") if isinstance(hints, dict) else None
        return {
            "classification": plan.get("classification") if isinstance(plan, dict) else "VECTOR_QUERY",
            "analysis_intent": hints.get("analysis_intent") if isinstance(hints, dict) else "standard_retrieval",
            "map_mode": "standard",
            "focus_label": hints.get("city") if isinstance(hints, dict) else None,
            "center_lat": float(coords["lat"]) if isinstance(coords, dict) and "lat" in coords else None,
            "center_lon": float(coords["lon"]) if isinstance(coords, dict) and "lon" in coords else None,
            "radius_km": float(hints["radius_km"]) if isinstance(hints, dict) and hints.get("radius_km") is not None else None,
            "counties": hints.get("counties") if isinstance(hints, dict) and isinstance(hints.get("counties"), list) else [],
            "gap_report": {},
            "result_count": 0,
            "county_coverage_count": 0,
        }

    def _run_sql_retrieval(self, question: str, hints: dict) -> pd.DataFrame:
        if hints.get("company_name") and str(hints.get("analysis_intent") or "") == "disruption_alternatives":
            return self.sql_engine.get_companies_by_name(str(hints["company_name"]))
        if hints.get("metric"):
            return self.sql_engine.get_top_companies_by_metric(str(hints["metric"]), limit=15)
        if hints.get("industry_group"):
            return self.sql_engine.get_companies_by_industry(str(hints["industry_group"]))
        if hints.get("oem") and not any(hints.get(key) for key in ["category_term", "capability_term", "city"]):
            return self.sql_engine.get_companies_by_oem(str(hints["oem"]))
        if any(hints.get(key) for key in ["oem", "category_term", "capability_term", "city", "company_name", "counties"]):
            return self.sql_engine.search_companies(
                oem_name=str(hints["oem"]) if hints.get("oem") else None,
                category_term=str(hints["category_term"]) if hints.get("category_term") else None,
                capability_term=str(hints["capability_term"]) if hints.get("capability_term") else None,
                city_term=str(hints["city"]) if hints.get("city") else None,
                county_names=hints.get("counties") if isinstance(hints.get("counties"), list) else None,
                company_term=str(hints["company_name"]) if hints.get("company_name") else None,
                limit=60,
            )
        if hints.get("company_name"):
            return self.sql_engine.get_companies_by_name(str(hints["company_name"]))

        lower = question.lower()
        if "top" in lower and "employment" in lower:
            return self.sql_engine.get_top_companies_by_metric("employment", limit=15)
        return pd.DataFrame()

    @staticmethod
    def _filter_by_oem(df: pd.DataFrame, oem: str) -> pd.DataFrame:
        if df.empty or "primary_oems" not in df.columns:
            return df
        mask = df["primary_oems"].fillna("").str.lower().str.contains(oem.lower())
        return df[mask].copy()

    @staticmethod
    def _optional_keyword_filter(df: pd.DataFrame, question: str) -> pd.DataFrame:
        if df.empty:
            return df

        lowered = question.lower()
        keyword_map = {
            "battery": ["battery"],
            "supplier": ["supplier", "supply"],
            "manufacturer": ["manufacturing", "manufacturer", "plant"],
            "manufacturers": ["manufacturing", "manufacturer", "plant"],
            "oem": ["oem"],
            "employment": ["employment", "employees"],
            "stamping": ["stamping"],
            "thermal": ["thermal", "hvac", "cooling"],
            "charging": ["charging", "charger"],
            "wiring": ["wiring", "harness", "connector"],
        }

        active_terms: List[str] = []
        for trigger, terms in keyword_map.items():
            if trigger in lowered:
                active_terms.extend(terms)

        if not active_terms:
            return df

        text_cols = [
            "chunk_text",
            "product_service",
            "ev_supply_chain_role",
            "industry_group",
            "primary_oems",
            "primary_facility_type",
            "category",
        ]
        existing_cols = [col for col in text_cols if col in df.columns]
        if not existing_cols:
            return df

        combined = df[existing_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        mask = combined.apply(lambda t: any(term in t for term in active_terms))
        filtered = df[mask].copy()
        return filtered if not filtered.empty else df

    @staticmethod
    def _apply_structured_filters(df: pd.DataFrame, hints: dict) -> pd.DataFrame:
        if df.empty:
            return df

        filtered = df.copy()
        if hints.get("category_term") and "category" in filtered.columns:
            mask = filtered["category"].fillna("").str.lower().str.contains(str(hints["category_term"]).lower())
            if mask.any():
                filtered = filtered[mask].copy()

        if hints.get("capability_term"):
            text_cols = [col for col in ["industry_group", "product_service", "ev_supply_chain_role", "chunk_text"] if col in filtered.columns]
            if text_cols:
                combined = filtered[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
                capability_tokens = [
                    token
                    for token in re.findall(r"[a-z0-9]+", str(hints["capability_term"]).lower())
                    if len(token) > 2
                ]
                mask = combined.apply(
                    lambda text: any(token in text for token in capability_tokens)
                ) if capability_tokens else combined.str.contains(str(hints["capability_term"]).lower())
                if mask.any():
                    filtered = filtered[mask].copy()

        if hints.get("city"):
            text_cols = [col for col in ["city", "location", "county"] if col in filtered.columns]
            if text_cols:
                combined = filtered[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
                mask = combined.str.contains(str(hints["city"]).lower())
                if mask.any():
                    filtered = filtered[mask].copy()

        if hints.get("counties") and "county" in filtered.columns and isinstance(hints.get("counties"), list):
            wanted = {str(county).strip().lower().replace(" county", "") for county in hints["counties"] if str(county).strip()}
            if wanted:
                county_series = (
                    filtered["county"].fillna("").astype(str).str.lower().str.replace(" county", "", regex=False).str.strip()
                )
                mask = county_series.isin(wanted)
                if mask.any():
                    filtered = filtered[mask].copy()

        return filtered

    @staticmethod
    def _merge_candidates(frames: List[pd.DataFrame]) -> pd.DataFrame:
        non_empty = [df for df in frames if not df.empty]
        if not non_empty:
            return pd.DataFrame()

        merged = pd.concat(non_empty, ignore_index=True, sort=False)
        dedupe_cols = [
            col
            for col in ["company", "location", "address"]
            if col in merged.columns
        ]
        if dedupe_cols:
            merged = merged.drop_duplicates(subset=dedupe_cols, keep="first")
        return merged

    @staticmethod
    def _choose_final_results(sql_df: pd.DataFrame, vector_df: pd.DataFrame, geo_df: pd.DataFrame) -> pd.DataFrame:
        if not geo_df.empty:
            return geo_df
        if not sql_df.empty and not vector_df.empty:
            return HybridGeospatialRAGPipeline._merge_candidates([sql_df, vector_df])
        if not sql_df.empty:
            return sql_df
        if not vector_df.empty:
            return vector_df
        return pd.DataFrame()

    @staticmethod
    def _df_to_records(df: pd.DataFrame) -> List[dict]:
        if df.empty:
            return []
        safe = df.where(pd.notnull(df), None).copy()
        return safe.to_dict(orient="records")

    @classmethod
    def _query_terms(cls, text: str) -> set[str]:
        tokens = set(re.findall(r"[a-z0-9]+", str(text).lower()))
        return {token for token in tokens if len(token) > 2 and token not in QUESTION_STOPWORDS}

    def _should_reject_as_unsupported(
        self,
        question: str,
        plan: Dict[str, object],
        sql_df: pd.DataFrame,
        vector_df: pd.DataFrame,
        geo_df: pd.DataFrame,
        final_df: pd.DataFrame,
    ) -> bool:
        if plan.get("geo"):
            return False
        if not sql_df.empty or not geo_df.empty:
            return False
        if vector_df.empty and final_df.empty:
            return True

        max_lexical = 0.0
        max_hybrid = 0.0
        if not vector_df.empty:
            if "lexical_score" in vector_df.columns:
                max_lexical = float(pd.to_numeric(vector_df["lexical_score"], errors="coerce").fillna(0.0).max())
            if "hybrid_score" in vector_df.columns:
                max_hybrid = float(pd.to_numeric(vector_df["hybrid_score"], errors="coerce").fillna(0.0).max())

        query_terms = self._query_terms(question)
        top_text = " ".join(vector_df.head(5).get("chunk_text", pd.Series(dtype=str)).fillna("").astype(str).tolist())
        overlap_terms = query_terms.intersection(self._query_terms(top_text))

        if max_lexical >= float(os.getenv("RAG_MIN_LEXICAL_SCORE", "0.15")):
            return False
        if overlap_terms:
            return False
        if self.vector_engine.embed_mode == "sentence-transformers" and max_hybrid >= float(
            os.getenv("RAG_MIN_SEMANTIC_HYBRID_SCORE", "0.72")
        ):
            return False
        return True

    def _unsupported_response(self, question: str, plan: Dict[str, object]) -> dict:
        return {
            "answer": (
                "Direct Answer\n"
                f"I could not find grounded evidence in the GNEM company dataset and Georgia county map to answer '{question}'.\n\n"
                "Spatial / Supply-Chain Details\n"
                "Ask about manufacturer locations, county-specific supplier clusters, distance/radius searches, "
                "OEM relationships, battery/component capabilities, or disruption alternatives in Georgia.\n\n"
                "Evidence Gaps\n"
                "No high-confidence SQL/vector/geo matches were available for this prompt."
            ),
            "sources": [],
            "retrieved_chunks": [],
            "retrieved_companies": [],
            "plan": plan,
            "map_context": self._build_initial_map_context(plan),
            "model_used": "not_called",
        }

    def _annotate_map_weights(self, df: pd.DataFrame, question: str, plan: Dict[str, object]) -> pd.DataFrame:
        if df.empty:
            return df

        out = df.copy()
        hints = plan.get("hints", {}) if isinstance(plan, dict) else {}
        if "distance_km" in out.columns and "distance_miles" not in out.columns:
            out["distance_miles"] = pd.to_numeric(out["distance_km"], errors="coerce") * 0.621371

        out["map_relevance"] = self._compute_relevance_component(out)
        out["map_query_match"] = self._compute_query_match_component(out, question=question)
        out["map_proximity"] = self._compute_proximity_component(out, hints=hints)
        out["map_metric"] = self._compute_metric_component(out, hints=hints)
        out["map_business_priority"] = out.apply(
            lambda row: self._business_priority_score(row, hints=hints),
            axis=1,
        )

        component_weights = {
            "map_relevance": 0.30,
            "map_query_match": 0.25,
            "map_proximity": 0.20,
            "map_business_priority": 0.15,
            "map_metric": 0.10,
        }

        map_weights: List[float] = []
        reasons: List[str] = []
        for _, row in out.iterrows():
            total_weight = 0.0
            weighted_sum = 0.0
            parts: List[str] = []
            for column, column_weight in component_weights.items():
                value = row.get(column)
                if pd.isna(value):
                    continue
                value_f = float(value)
                weighted_sum += value_f * column_weight
                total_weight += column_weight
                if column == "map_relevance" and value_f > 0:
                    parts.append(f"retrieval={value_f:.2f}")
                elif column == "map_query_match" and value_f > 0:
                    parts.append(f"query_match={value_f:.2f}")
                elif column == "map_proximity" and value_f > 0:
                    parts.append(f"proximity={value_f:.2f}")
                elif column == "map_business_priority" and value_f > 0:
                    parts.append(f"business={value_f:.2f}")
                elif column == "map_metric" and value_f > 0:
                    parts.append(f"metric={value_f:.2f}")

            score = weighted_sum / total_weight if total_weight > 0 else 0.5
            map_weights.append(round(max(0.05, min(1.0, score)), 4))
            reasons.append(", ".join(parts) if parts else "default=0.50")

        out["map_weight"] = map_weights
        out["map_weight_reason"] = reasons
        if "score" not in out.columns:
            out["score"] = out["map_weight"]
        else:
            out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(out["map_weight"])
        return out

    def _annotate_peer_distances(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or not {"latitude", "longitude", "company"}.issubset(df.columns) or len(df) < 2:
            return df

        out = df.copy().reset_index(drop=True)
        nearest_companies: List[Optional[str]] = []
        nearest_km_values: List[Optional[float]] = []

        for idx, row in out.iterrows():
            lat = row.get("latitude")
            lon = row.get("longitude")
            if pd.isna(lat) or pd.isna(lon):
                nearest_companies.append(None)
                nearest_km_values.append(None)
                continue

            others = out.drop(index=idx)
            others = others.dropna(subset=["latitude", "longitude"]).copy()
            distinct_others = others[others["company"].astype(str) != str(row.get("company"))].copy()
            if not distinct_others.empty:
                others = distinct_others
            if others.empty:
                nearest_companies.append(None)
                nearest_km_values.append(None)
                continue

            distances = self.spatial_engine._haversine_km(
                lat1=float(lat),
                lon1=float(lon),
                lat2_series=others["latitude"],
                lon2_series=others["longitude"],
            )
            best_idx = distances.idxmin()
            nearest_companies.append(str(others.loc[best_idx, "company"]))
            nearest_km = float(distances.loc[best_idx])
            nearest_km_values.append(round(nearest_km, 3))

        out["nearest_peer_company"] = nearest_companies
        out["nearest_peer_distance_km"] = nearest_km_values
        out["nearest_peer_distance_miles"] = pd.to_numeric(
            out["nearest_peer_distance_km"], errors="coerce"
        ) * 0.621371
        return out

    @staticmethod
    def _compute_relevance_component(df: pd.DataFrame) -> pd.Series:
        if "hybrid_score" in df.columns:
            series = pd.to_numeric(df["hybrid_score"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
            return series
        if "lexical_score" in df.columns:
            return pd.to_numeric(df["lexical_score"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
        if "semantic_score" in df.columns:
            semantic = pd.to_numeric(df["semantic_score"], errors="coerce").fillna(0.0)
            return ((semantic + 1.0) / 2.0).clip(lower=0.0, upper=1.0)
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")

    def _compute_query_match_component(self, df: pd.DataFrame, question: str) -> pd.Series:
        query_terms = self._query_terms(question)
        if not query_terms:
            return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")

        text_cols = [
            col
            for col in [
                "company",
                "category",
                "industry_group",
                "ev_supply_chain_role",
                "primary_oems",
                "supplier_or_affiliation_type",
                "product_service",
                "primary_facility_type",
                "chunk_text",
                "city",
                "county",
            ]
            if col in df.columns
        ]
        if not text_cols:
            return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")

        combined = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        scores = []
        for text in combined:
            row_terms = self._query_terms(text)
            overlap = len(query_terms.intersection(row_terms))
            scores.append(overlap / max(1, len(query_terms)))
        return pd.Series(scores, index=df.index, dtype="float64").clip(lower=0.0, upper=1.0)

    @staticmethod
    def _compute_proximity_component(df: pd.DataFrame, hints: Dict[str, object]) -> pd.Series:
        if "distance_km" not in df.columns:
            return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")

        distances = pd.to_numeric(df["distance_km"], errors="coerce")
        if distances.notna().sum() == 0:
            return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")

        radius_hint = float(hints.get("radius_km", 0.0) or 0.0)
        max_distance = float(distances.max(skipna=True) or 0.0)
        scale = max(radius_hint, max_distance, 1.0)
        proximity = 1.0 - (distances / scale)
        return proximity.fillna(pd.NA).clip(lower=0.0, upper=1.0)

    @staticmethod
    def _compute_metric_component(df: pd.DataFrame, hints: Dict[str, object]) -> pd.Series:
        metric_col = "metric_value" if "metric_value" in df.columns else None
        if metric_col is None:
            return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")

        values = pd.to_numeric(df[metric_col], errors="coerce")
        valid = values.dropna()
        if valid.empty:
            return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
        if float(valid.max()) == float(valid.min()):
            return pd.Series([1.0] * len(df), index=df.index, dtype="float64")

        normalized = (values - float(valid.min())) / max(float(valid.max()) - float(valid.min()), 1e-6)
        if hints.get("metric"):
            return normalized.fillna(pd.NA).clip(lower=0.0, upper=1.0)
        return normalized.fillna(pd.NA).clip(lower=0.0, upper=1.0)

    @staticmethod
    def _business_priority_score(row: pd.Series, hints: Dict[str, object]) -> float:
        category = str(row.get("category") or "").lower()
        role = str(row.get("ev_supply_chain_role") or "").lower()
        affiliation = str(row.get("supplier_or_affiliation_type") or "").lower()
        facility = str(row.get("primary_facility_type") or "").lower()
        product = str(row.get("product_service") or "").lower()
        oems = str(row.get("primary_oems") or "").lower()
        combined = " ".join([category, role, affiliation, facility, product, oems])

        score = 0.35
        if "oem" in category or "oem" in role:
            score = max(score, 0.85)
        elif "tier 1" in category or "tier 1" in role:
            score = max(score, 0.75)
        elif "tier 2" in category or "tier 2" in role:
            score = max(score, 0.68)
        elif "supplier" in role or "supplier" in affiliation:
            score = max(score, 0.60)

        if any(term in facility for term in ["manufact", "plant", "assembly", "stamping"]):
            score += 0.10
        if any(term in facility for term in ["warehouse", "distribution", "logistics"]):
            score += 0.05

        category_term = str(hints.get("category_term") or "").lower()
        capability_term = str(hints.get("capability_term") or "").lower()
        oem_term = str(hints.get("oem") or "").lower()
        if category_term and category_term in category:
            score += 0.15
        capability_tokens = [token for token in re.findall(r"[a-z0-9]+", capability_term) if len(token) > 2]
        if capability_tokens and any(token in combined for token in capability_tokens):
            score += 0.15
        if oem_term and oem_term in oems:
            score += 0.15

        return round(max(0.0, min(1.0, score)), 4)

    @staticmethod
    def _build_geo_no_results_chunk(question: str, hints: dict) -> Dict[str, object]:
        radius_km = hints.get("radius_km")
        radius_text = f"{float(radius_km):.1f} km" if radius_km is not None else "the requested radius"
        if hints.get("company_name"):
            target = f"around {hints['company_name']}"
        elif hints.get("city"):
            target = f"near {hints['city']}"
        elif hints.get("counties"):
            target = f"in counties {', '.join(str(c) for c in hints['counties'])}"
        elif hints.get("coordinates"):
            coords = hints["coordinates"]
            target = f"near coordinates ({coords.get('lat')}, {coords.get('lon')})"
        else:
            target = "for the requested geography"

        filters = []
        if hints.get("oem"):
            filters.append(f"OEM={hints['oem']}")
        if hints.get("category_term"):
            filters.append(f"category={hints['category_term']}")
        if hints.get("capability_term"):
            filters.append(f"capability={hints['capability_term']}")
        filter_text = f" after applying filters ({', '.join(filters)})" if filters else ""

        return {
            "chunk_id": "C1",
            "engine": "geo",
            "company": None,
            "chunk_type": "geo_no_results",
            "score": 1.0,
            "text": (
                f"No companies matched {radius_text} {target}{filter_text}. "
                f"Question searched: {question}"
            ),
            "meta": {"radius_km": radius_km, "target": target, "filters": filters},
        }

    def _build_retrieved_chunks(
        self,
        question: str,
        vector_df: pd.DataFrame,
        sql_df: pd.DataFrame,
        geo_df: pd.DataFrame,
        final_df: pd.DataFrame,
        gap_report: Optional[Dict[str, object]] = None,
        plan: Optional[Dict[str, object]] = None,
    ) -> List[Dict[str, object]]:
        chunks: List[Dict[str, object]] = []
        final_companies = set(final_df["company"].dropna().astype(str).tolist()) if "company" in final_df.columns else set()
        hints = plan.get("hints", {}) if isinstance(plan, dict) else {}
        analysis_intent = str(hints.get("analysis_intent") or "standard_retrieval")

        if gap_report and gap_report.get("gap_county_count"):
            top_gap_names = [
                str(item.get("county"))
                for item in gap_report.get("gap_counties", [])[:6]
                if item.get("county")
            ]
            chunks.append(
                {
                    "engine": "geo",
                    "company": None,
                    "chunk_type": "coverage_gap_summary",
                    "score": 1.2,
                    "text": (
                        f"Gap analysis found {gap_report.get('gap_county_count')} uncovered counties "
                        f"out of {gap_report.get('scope_county_count')} in scope. "
                        f"Highest-distance gap counties include: {', '.join(top_gap_names) if top_gap_names else 'N/A'}."
                    ),
                    "meta": {
                        "gap_county_count": gap_report.get("gap_county_count"),
                        "covered_county_count": gap_report.get("covered_county_count"),
                        "top_gap_counties": gap_report.get("gap_counties", [])[:8],
                    },
                    "priority": 0,
                }
            )

        if not vector_df.empty:
            for _, row in vector_df.head(8).iterrows():
                company_name = str(row.get("company") or "").strip()
                if final_companies and company_name and company_name not in final_companies:
                    continue
                chunks.append(
                    {
                        "engine": "vector",
                        "company": row.get("company"),
                        "chunk_type": row.get("chunk_type", "vector_chunk"),
                        "score": float(row.get("hybrid_score", row.get("semantic_score", 0.0))),
                        "text": str(row.get("chunk_text", "")).strip(),
                        "meta": {
                            "semantic_score": row.get("semantic_score"),
                            "lexical_score": row.get("lexical_score"),
                            "chunk_ref": row.get("chunk_id"),
                        },
                        "priority": 1 if str(row.get("company")) in final_companies else 2,
                    }
                )

        if not geo_df.empty:
            for rank, (_, row) in enumerate(geo_df.head(6).iterrows(), start=1):
                distance_km = row.get("distance_km")
                dist_txt = f"{float(distance_km):.2f} km / {float(distance_km) * 0.621371:.2f} mi" if pd.notna(distance_km) else "unknown"
                if analysis_intent == "disruption_alternatives" and row.get("disrupted_company"):
                    text = (
                        f"Alternative supplier candidate for {row.get('disrupted_company')}: {row.get('company')} "
                        f"in {row.get('city')}, {row.get('county')} at {dist_txt}. "
                        f"Match reason: {row.get('match_reason') or 'nearby supplier candidate'}. "
                        f"Product/service: {row.get('product_service') or 'unknown'}."
                    )
                    chunk_type = "alternative_supplier"
                else:
                    text = (
                        f"Geo match for {row.get('company')}: "
                        f"{row.get('city')}, {row.get('county')} at distance {dist_txt}; "
                        f"lat={row.get('latitude')}, lon={row.get('longitude')}. "
                        f"Role={row.get('ev_supply_chain_role')}; Product={row.get('product_service')}."
                    )
                    chunk_type = "geo_match"
                chunks.append(
                    {
                        "engine": "geo",
                        "company": row.get("company"),
                        "chunk_type": chunk_type,
                        "score": 1.0 / rank,
                        "text": text,
                        "meta": {
                            "distance_km": distance_km,
                            "distance_miles": row.get("distance_miles"),
                            "match_reason": row.get("match_reason"),
                            "disrupted_company": row.get("disrupted_company"),
                        },
                        "priority": 1,
                    }
                )

        if not sql_df.empty:
            for rank, (_, row) in enumerate(sql_df.head(6).iterrows(), start=1):
                company_name = str(row.get("company") or "").strip()
                if final_companies and company_name and company_name not in final_companies:
                    continue
                metric_value = row.get("metric_value", row.get("employment"))
                text = (
                    f"SQL result for {row.get('company')}: "
                    f"industry={row.get('industry_group')}, role={row.get('ev_supply_chain_role')}, "
                    f"OEMs={row.get('primary_oems')}, metric={metric_value}."
                )
                chunks.append(
                    {
                        "engine": "sql",
                        "company": row.get("company"),
                        "chunk_type": "sql_result",
                        "score": 1.0 / rank,
                        "text": text,
                        "meta": {"metric_value": metric_value},
                        "priority": 1 if str(row.get("company")) in final_companies else 3,
                    }
                )

        chunks = self._dedupe_chunks(chunks)
        chunks = sorted(chunks, key=lambda x: (x["priority"], -float(x.get("score", 0.0))))
        if not chunks:
            chunks = [
                {
                    "engine": "system",
                    "company": None,
                    "chunk_type": "no_results",
                    "score": 0.0,
                    "text": f"No retrieval hits were found for question: {question}",
                    "meta": {},
                    "priority": 9,
                }
            ]

        out: List[Dict[str, object]] = []
        max_chunks = 6 if final_companies else 8
        for idx, chunk in enumerate(chunks[:max_chunks], start=1):
            out.append(
                {
                    "chunk_id": f"C{idx}",
                    "engine": chunk["engine"],
                    "company": chunk.get("company"),
                    "chunk_type": chunk.get("chunk_type"),
                    "score": round(float(chunk.get("score", 0.0)), 4),
                    "text": str(chunk.get("text", "")),
                    "meta": chunk.get("meta", {}),
                }
            )
        return out

    @staticmethod
    def _dedupe_chunks(chunks: List[Dict[str, object]]) -> List[Dict[str, object]]:
        seen = set()
        out = []
        for chunk in chunks:
            key = (
                chunk.get("engine"),
                chunk.get("company"),
                str(chunk.get("chunk_type")),
                str(chunk.get("text", ""))[:180],
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(chunk)
        return out

    def _format_context(
        self,
        question: str,
        plan: dict,
        retrieved_chunks: List[Dict[str, object]],
        final_df: Optional[pd.DataFrame] = None,
        map_context: Optional[Dict[str, object]] = None,
    ) -> str:
        lines = [
            f"Question: {question}",
            f"Plan Classification: {plan.get('classification')}",
            f"Analysis Intent: {(plan.get('hints', {}) or {}).get('analysis_intent', 'standard_retrieval')}",
        ]
        if map_context:
            lines.append(
                "Map Context: "
                f"mode={map_context.get('map_mode')}, "
                f"focus={map_context.get('focus_label')}, "
                f"center=({map_context.get('center_lat')}, {map_context.get('center_lon')}), "
                f"radius_km={map_context.get('radius_km')}, "
                f"counties={map_context.get('counties')}, "
                f"result_count={map_context.get('result_count')}, "
                f"county_coverage_count={map_context.get('county_coverage_count')}"
            )
            gap_report = map_context.get("gap_report") if isinstance(map_context.get("gap_report"), dict) else {}
            if gap_report:
                lines.append(
                    "Gap Report: "
                    f"covered_counties={gap_report.get('covered_county_count')}, "
                    f"gap_counties={gap_report.get('gap_county_count')}, "
                    f"top_gap_counties={gap_report.get('gap_counties', [])[:5]}"
                )

        if final_df is not None and not final_df.empty:
            lines.append("Top Retrieved Companies:")
            show_cols = [
                col
                for col in [
                    "company",
                    "city",
                    "county",
                    "category",
                    "ev_supply_chain_role",
                    "product_service",
                    "distance_km",
                    "match_reason",
                ]
                if col in final_df.columns
            ]
            for _, row in final_df.head(8).iterrows():
                parts = []
                for col in show_cols:
                    value = row.get(col)
                    if pd.notna(value) and str(value).strip():
                        if col == "distance_km":
                            parts.append(f"{col}={float(value):.2f}")
                        else:
                            parts.append(f"{col}={str(value).strip()}")
                if parts:
                    lines.append(" - " + " | ".join(parts))
        lines.append("Retrieved Chunks:")
        for chunk in retrieved_chunks:
            company = chunk.get("company") or "N/A"
            chunk_text = str(chunk.get("text", "")).strip()
            if len(chunk_text) > 140:
                chunk_text = chunk_text[:140] + "..."
            lines.append(
                f"[{chunk['chunk_id']}] engine={chunk['engine']} | company={company} | "
                f"type={chunk['chunk_type']} | score={chunk['score']}"
            )
            lines.append(chunk_text)
        return "\n".join(lines).strip()

    @staticmethod
    def _chunk_source_line(chunk: Dict[str, object]) -> str:
        snippet = str(chunk.get("text", "")).replace("\n", " ").strip()
        if len(snippet) > 140:
            snippet = snippet[:140] + "..."
        return (
            f"[{chunk.get('chunk_id')}] {chunk.get('engine')} | "
            f"{chunk.get('company') or 'N/A'} | score={chunk.get('score')} | {snippet}"
        )

    def _generate_answer_with_llm(
        self,
        question: str,
        context: str,
        retrieved_chunks: List[Dict[str, object]],
        preferred_companies: Optional[List[str]] = None,
    ) -> str:
        self.available_models = self._list_available_models()
        if not self.available_models and self.llm_model == "retrieval-fallback":
            return self._fast_fallback_answer(
                question=question,
                retrieved_chunks=retrieved_chunks,
                error=None,
                preferred_companies=preferred_companies or [],
            )
        system_prompt = (
            "You are a production geospatial EV supply-chain intelligence analyst. "
            "You MUST answer only from the retrieved chunks and cite chunk IDs like [C3]. "
            "If evidence is missing or ambiguous, say so clearly. "
            "Prefer concise operational analysis with distances, counties, supplier roles, and backup options."
        )
        user_prompt = (
            f"{context}\n\n"
            "Instructions:\n"
            "1. Write 3 compact sections with these headings exactly: Direct Answer, Spatial / Supply-Chain Details, Evidence Gaps.\n"
            "2. Include distances in both km and miles when distance evidence exists.\n"
            "3. For disruption or gap questions, identify operational implications and nearby alternatives or uncovered counties.\n"
            "4. Include at least 2 chunk citations when evidence exists.\n"
            "5. Do not fabricate company names, distances, OEM links, or metrics.\n"
            "6. Do not repeat the user question.\n"
            "7. Do not use markdown heading symbols like #, ##, or ###."
        )
        model_candidates = [self.llm_model] + self._oom_fallback_candidates(self.llm_model)
        max_model_attempts = int(os.getenv("OLLAMA_MAX_MODEL_ATTEMPTS", "2"))
        model_candidates = model_candidates[:max(1, max_model_attempts)]
        last_exc: Optional[Exception] = None
        request_timeout = float(os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "60"))
        max_output_tokens = int(os.getenv("OLLAMA_MAX_OUTPUT_TOKENS", "900"))
        for model_name in model_candidates:
            try:
                response = self.llm_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=max_output_tokens,
                    timeout=request_timeout,
                    extra_body={"options": {"num_predict": max_output_tokens, "num_ctx": 4096}},
                )
                content = self._normalize_message_text(
                    response.choices[0].message.content if response.choices else None
                )
                if not content:
                    raise RuntimeError(f"Ollama returned an empty response for model '{model_name}'.")
                self.llm_model = model_name
                return content.strip()
            except Exception as exc:
                last_exc = exc
                if (
                    self._is_memory_error(exc)
                    or self._is_model_unavailable_error(exc)
                    or self._is_timeout_error(exc)
                    or self._is_empty_response_error(exc)
                ):
                    continue

        return self._fast_fallback_answer(
            question=question,
            retrieved_chunks=retrieved_chunks,
            error=last_exc,
            preferred_companies=preferred_companies or [],
        )

    def _oom_fallback_candidates(self, current_model: str) -> List[str]:
        available = [m for m in self.available_models if m != current_model]
        ordered: List[str] = []

        for pref in self.fallback_model_preferences:
            if pref in available and pref not in ordered:
                ordered.append(pref)

        current_size = self._model_size_b(current_model)
        dynamic = sorted(available, key=self._model_sort_key)
        for candidate in dynamic:
            candidate_size = self._model_size_b(candidate)
            if current_size is None or candidate_size is None or candidate_size < current_size:
                if candidate not in ordered:
                    ordered.append(candidate)
        return ordered

    @staticmethod
    def _model_size_b(model_name: str) -> Optional[float]:
        match = re.search(r"(\d+(?:\.\d+)?)b", model_name.lower())
        if not match:
            return None
        return float(match.group(1))

    @classmethod
    def _model_sort_key(cls, model_name: str) -> tuple:
        size = cls._model_size_b(model_name)
        return (size is None, float(size) if size is not None else 10_000.0, model_name)

    @staticmethod
    def _normalize_message_text(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                    parts.append(str(item["text"]))
                elif hasattr(item, "text") and getattr(item, "text"):
                    parts.append(str(getattr(item, "text")))
            return "".join(parts)
        return str(value)

    @staticmethod
    def _is_memory_error(exc: Exception) -> bool:
        text = str(exc).lower()
        memory_signals = [
            "requires more system memory",
            "out of memory",
            "insufficient memory",
            "cuda out of memory",
        ]
        return any(signal in text for signal in memory_signals)

    @staticmethod
    def _is_model_unavailable_error(exc: Exception) -> bool:
        text = str(exc).lower()
        signals = [
            "model not found",
            "pull model",
            "does not exist",
        ]
        return any(signal in text for signal in signals)

    @staticmethod
    def _is_timeout_error(exc: Exception) -> bool:
        text = str(exc).lower()
        signals = [
            "timed out",
            "readtimeout",
            "apitimeouterror",
        ]
        return any(signal in text for signal in signals)

    @staticmethod
    def _is_empty_response_error(exc: Exception) -> bool:
        text = str(exc).lower()
        signals = [
            "empty response",
            "reasoning-only response",
        ]
        return any(signal in text for signal in signals)

    @staticmethod
    def _fast_fallback_answer(
        question: str,
        retrieved_chunks: List[Dict[str, object]],
        error: Optional[Exception],
        preferred_companies: Optional[List[str]] = None,
    ) -> str:
        if retrieved_chunks:
            first_chunk = retrieved_chunks[0]
            has_alternative_chunks = any(
                chunk.get("chunk_type") == "alternative_supplier" and chunk.get("company")
                for chunk in retrieved_chunks
            )
            if first_chunk.get("chunk_type") == "coverage_gap_summary" and not has_alternative_chunks:
                return (
                    "Direct Answer\n"
                    f"{first_chunk.get('text')} [C1]\n\n"
                    "Spatial / Supply-Chain Details\n"
                    "Use the red county shading on the map and the retrieved supplier table to inspect where capability coverage is sparse or absent.\n\n"
                    "Evidence Gaps\n"
                    "Ollama response timed out or failed, so this answer uses the computed gap summary and retrieved records only."
                )
            if first_chunk.get("chunk_type") == "geo_no_results":
                return (
                    "Direct Answer\n"
                    f"{first_chunk.get('text')} [C1]\n\n"
                    "Spatial / Supply-Chain Details\n"
                    "No facilities matched the requested spatial and business filters in the indexed Georgia dataset.\n\n"
                    "Evidence Gaps\n"
                    "Ollama response timed out or failed, so this answer is based on retrieval only."
                )
            if first_chunk.get("chunk_type") == "no_results":
                return (
                    "Direct Answer\n"
                    f"{first_chunk.get('text')} [C1]\n\n"
                    "Spatial / Supply-Chain Details\n"
                    "The retrieval layer did not return company, location, or supply-chain records for this question.\n\n"
                    "Evidence Gaps\n"
                    "Ollama response timed out or failed, and retrieval had no supporting records."
                )

        companies: List[str] = []
        citations: List[str] = []
        preferred = [str(company) for company in (preferred_companies or []) if company]
        if preferred:
            companies = preferred[:4]
            for chunk in retrieved_chunks:
                cid = chunk.get("chunk_id")
                if chunk.get("company") in companies and cid:
                    citations.append(f"[{cid}]")
                if len(citations) >= 2:
                    break

        if not companies:
            for chunk in retrieved_chunks:
                company = chunk.get("company")
                cid = chunk.get("chunk_id")
                if company and company not in companies:
                    companies.append(str(company))
                if cid:
                    citations.append(f"[{cid}]")
                if len(companies) >= 4 and len(citations) >= 2:
                    break

        if companies:
            refs = " ".join(citations[:2]) if citations else ""
            items = ", ".join(companies[:4])
            geo_details = []
            for chunk in retrieved_chunks:
                if chunk.get("engine") != "geo" or not chunk.get("company"):
                    continue
                meta = chunk.get("meta", {}) if isinstance(chunk.get("meta", {}), dict) else {}
                distance_km = meta.get("distance_km")
                if distance_km is not None:
                    geo_details.append(
                        f"{chunk.get('company')} at {float(distance_km):.1f} km / {float(distance_km) * 0.621371:.1f} mi"
                    )
                elif chunk.get("chunk_type") == "alternative_supplier":
                    geo_details.append(str(chunk.get("text", "")).split(".")[0])
                if len(geo_details) >= 3:
                    break
            spatial_detail = "; ".join(geo_details) if geo_details else "Open the map and table to compare counties, distances, and company roles for these facilities."
            return (
                "Direct Answer\n"
                f"Top matches based on retrieved evidence for '{question}': {items}. {refs}\n\n"
                "Spatial / Supply-Chain Details\n"
                f"{spatial_detail}\n\n"
                "Evidence Gaps\n"
                "Ollama response timed out or failed, so this is a fast retrieval-only fallback summary."
            )

        return (
            "Direct Answer\n"
            f"I could not generate a model answer in time for '{question}'.\n\n"
            "Spatial / Supply-Chain Details\n"
            "No high-confidence chunks were available to summarize from the indexed GNEM sources.\n\n"
            "Evidence Gaps\n"
            "Backend fell back because Ollama timed out or returned no usable text."
        )
