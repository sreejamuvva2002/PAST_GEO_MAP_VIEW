from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import duckdb
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim

from backend.ingestion import load_county_centroids

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "gnem.duckdb"
DEFAULT_GEOJSON_PATH = PROJECT_ROOT / "data" / "Counties_Georgia.geojson"


class SpatialEngine:
    CITY_TO_COUNTY = {
        "atlanta": "fulton",
        "savannah": "chatham",
        "augusta": "richmond",
        "macon": "bibb",
        "columbus": "muscogee",
        "athens": "clarke",
        "warner robins": "houston",
        "rome": "floyd",
        "valdosta": "lowndes",
        "albany": "dougherty",
        "johns creek": "fulton",
        "alpharetta": "fulton",
        "marietta": "cobb",
        "roswell": "fulton",
        "sandy springs": "fulton",
        "west point": "troup",
        "bainbridge": "decatur",
        "statesboro": "bulloch",
    }

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        geojson_path: Path = DEFAULT_GEOJSON_PATH,
    ) -> None:
        self.db_path = Path(db_path)
        self.geojson_path = Path(geojson_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"DuckDB file not found: {self.db_path}")

        self.companies_df = self._load_companies()
        self.city_centroids = self._build_city_centroids(self.companies_df)
        self.county_centroids = load_county_centroids(self.geojson_path) if self.geojson_path.exists() else {}
        self.county_names = sorted({county.title() for county in self.county_centroids.keys()})
        self.enable_external_geocoding = os.getenv("ENABLE_EXTERNAL_GEOCODING", "false").strip().lower() == "true"

    def _load_companies(self) -> pd.DataFrame:
        with duckdb.connect(str(self.db_path), read_only=True) as con:
            df = con.execute(
                """
                SELECT
                    company, category, industry_group, location, address, city, county,
                    ev_supply_chain_role, primary_oems, supplier_or_affiliation_type,
                    employment, product_service, ev_battery_relevant, primary_facility_type,
                    latitude, longitude, coordinate_source
                FROM companies
                """
            ).fetchdf()
        return df

    @staticmethod
    def _build_city_centroids(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        valid = df.dropna(subset=["city", "latitude", "longitude"]).copy()
        if valid.empty:
            return {}

        grouped = (
            valid.groupby(valid["city"].str.lower().str.strip())
            .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"))
            .reset_index()
        )

        out: Dict[str, Tuple[float, float]] = {}
        for _, row in grouped.iterrows():
            out[str(row["city"])] = (float(row["latitude"]), float(row["longitude"]))
        return out

    @staticmethod
    def _normalize_text(value: object) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    @staticmethod
    def _haversine_km(
        lat1: float,
        lon1: float,
        lat2_series: pd.Series,
        lon2_series: pd.Series,
    ) -> pd.Series:
        lat1_rad = math.radians(float(lat1))
        lon1_rad = math.radians(float(lon1))
        lat2 = np.radians(pd.to_numeric(lat2_series, errors="coerce").astype(float).to_numpy())
        lon2 = np.radians(pd.to_numeric(lon2_series, errors="coerce").astype(float).to_numpy())
        dlat = lat2 - lat1_rad
        dlon = lon2 - lon1_rad
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        return pd.Series(6371.0088 * c, index=lat2_series.index, dtype="float64")

    def resolve_place_coordinates(self, place_name: str) -> Optional[Tuple[float, float]]:
        return self._resolve_city_coordinates(place_name)

    def _resolve_city_coordinates(self, city_name: str) -> Optional[Tuple[float, float]]:
        key = city_name.strip().lower()
        if key in self.city_centroids:
            return self.city_centroids[key]

        if key.endswith(" county"):
            key = key.replace(" county", "").strip()
        if key in self.county_centroids:
            return self.county_centroids[key]

        mapped_county = self.CITY_TO_COUNTY.get(key)
        if mapped_county and mapped_county in self.county_centroids:
            return self.county_centroids[mapped_county]

        for city_key, coords in self.city_centroids.items():
            if key in city_key or city_key in key:
                return coords

        for county_name, coords in self.county_centroids.items():
            if key == county_name or key == f"{county_name} county":
                return coords

        if self.enable_external_geocoding:
            try:
                geocoder = Nominatim(user_agent="hybrid-geospatial-rag")
                location = geocoder.geocode(f"{city_name}, Georgia, USA", timeout=5)
                if location:
                    return float(location.latitude), float(location.longitude)
            except Exception:
                return None

        return None

    def companies_in_counties(
        self,
        counties: Sequence[str],
        candidates: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        base = candidates.copy() if candidates is not None and not candidates.empty else self.companies_df.copy()
        if base.empty or "county" not in base.columns:
            return pd.DataFrame(columns=list(self.companies_df.columns))

        wanted = {str(county).strip().lower().replace(" county", "") for county in counties if str(county).strip()}
        if not wanted:
            return pd.DataFrame(columns=list(base.columns))

        county_series = base["county"].fillna("").astype(str).str.lower().str.replace(" county", "", regex=False).str.strip()
        out = base[county_series.isin(wanted)].copy()
        if out.empty:
            return pd.DataFrame(columns=list(base.columns))
        out = out.sort_values(["county", "company", "location"], ascending=[True, True, True]).reset_index(drop=True)
        return out

    def companies_within_radius(
        self,
        lat: float,
        lon: float,
        radius_km: float,
        candidates: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        base = candidates.copy() if candidates is not None and not candidates.empty else self.companies_df.copy()
        base = base.dropna(subset=["latitude", "longitude"]).copy()
        if base.empty:
            return base

        base["distance_km"] = self._haversine_km(
            lat1=float(lat),
            lon1=float(lon),
            lat2_series=base["latitude"],
            lon2_series=base["longitude"],
        )
        base["distance_miles"] = base["distance_km"] * 0.621371
        within = base[base["distance_km"] <= float(radius_km)].copy()
        within = within.sort_values(["distance_km", "company"], ascending=[True, True]).reset_index(drop=True)
        return within

    def companies_near_city(
        self,
        city_name: str,
        radius_km: float = 50.0,
        candidates: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        coords = self._resolve_city_coordinates(city_name)
        if not coords:
            return pd.DataFrame(columns=list(self.companies_df.columns) + ["distance_km"])

        lat, lon = coords
        return self.companies_within_radius(lat, lon, radius_km=radius_km, candidates=candidates)

    def rank_alternative_suppliers(
        self,
        company_name: str,
        radius_km: float = 250.0,
        category_term: Optional[str] = None,
        capability_term: Optional[str] = None,
        max_results: int = 20,
    ) -> pd.DataFrame:
        key = self._normalize_text(company_name)
        if not key:
            return pd.DataFrame(columns=list(self.companies_df.columns) + ["distance_km", "distance_miles", "match_reason"])
        origin = self.companies_df[
            self.companies_df["company"].fillna("").astype(str).str.lower().str.contains(re.escape(key), regex=True)
        ].dropna(subset=["latitude", "longitude"])
        if origin.empty:
            return pd.DataFrame(columns=list(self.companies_df.columns) + ["distance_km", "distance_miles", "match_reason"])

        origin_row = origin.iloc[0]
        center_lat = float(origin_row["latitude"])
        center_lon = float(origin_row["longitude"])
        origin_company = str(origin_row["company"])
        target_category = self._normalize_text(category_term or origin_row.get("category"))
        target_capability = self._normalize_text(
            capability_term
            or origin_row.get("ev_supply_chain_role")
            or origin_row.get("industry_group")
            or origin_row.get("product_service")
        )

        candidates = self.companies_df.copy()
        candidates = candidates[candidates["company"].fillna("").astype(str) != origin_company].copy()
        combined = (
            candidates[
                [
                    "category",
                    "industry_group",
                    "ev_supply_chain_role",
                    "product_service",
                    "supplier_or_affiliation_type",
                    "primary_oems",
                ]
            ]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
        )
        if target_category:
            category_mask = candidates["category"].fillna("").astype(str).str.lower().str.contains(
                re.escape(target_category), regex=True
            )
        else:
            category_mask = pd.Series([False] * len(candidates), index=candidates.index)

        capability_terms = [
            token
            for token in re.findall(r"[a-z0-9]+", target_capability)
            if len(token) > 3 and token not in {"automotive", "manufacturing", "systems", "components", "supplier"}
        ]
        if capability_terms:
            capability_mask = combined.apply(lambda text: any(term in text for term in capability_terms))
        else:
            capability_mask = pd.Series([False] * len(candidates), index=candidates.index)

        candidates = candidates[category_mask | capability_mask].copy()
        if candidates.empty:
            candidates = self.companies_df[
                self.companies_df["company"].fillna("").astype(str) != origin_company
            ].copy()

        ranked = self.companies_within_radius(
            lat=center_lat,
            lon=center_lon,
            radius_km=radius_km,
            candidates=candidates,
        ).copy()
        if ranked.empty:
            return ranked

        ranked["match_reason"] = ranked.apply(
            lambda row: self._supplier_match_reason(row, target_category, capability_terms, origin_company),
            axis=1,
        )
        ranked["disrupted_company"] = origin_company
        ranked["query_center_lat"] = center_lat
        ranked["query_center_lon"] = center_lon
        return ranked.head(int(max_results)).reset_index(drop=True)

    @staticmethod
    def _supplier_match_reason(
        row: pd.Series,
        target_category: str,
        capability_terms: Sequence[str],
        origin_company: str,
    ) -> str:
        pieces = []
        if target_category and target_category in str(row.get("category") or "").lower():
            pieces.append(f"same category as {origin_company}")
        haystack = " ".join(
            str(row.get(col) or "")
            for col in ["industry_group", "ev_supply_chain_role", "product_service", "primary_oems"]
        ).lower()
        matched_terms = [term for term in capability_terms if term in haystack][:4]
        if matched_terms:
            pieces.append("capability overlap: " + ", ".join(matched_terms))
        if not pieces:
            pieces.append("nearby supplier candidate")
        return "; ".join(pieces)

    def supply_gap_report(
        self,
        capability_term: Optional[str] = None,
        category_term: Optional[str] = None,
        county_scope: Optional[Sequence[str]] = None,
        max_gap_counties: int = 12,
    ) -> Dict[str, object]:
        df = self.companies_df.dropna(subset=["county", "latitude", "longitude"]).copy()
        if category_term:
            mask = df["category"].fillna("").astype(str).str.lower().str.contains(
                re.escape(str(category_term).lower()), regex=True
            )
            if mask.any():
                df = df[mask].copy()
        if capability_term:
            combined = (
                df[["industry_group", "product_service", "ev_supply_chain_role"]]
                .fillna("")
                .astype(str)
                .agg(" ".join, axis=1)
                .str.lower()
            )
            mask = combined.str.contains(re.escape(str(capability_term).lower()), regex=True)
            if mask.any():
                df = df[mask].copy()

        covered = {str(county).strip().title() for county in df["county"].dropna().astype(str).tolist() if str(county).strip()}
        scoped_counties = (
            {str(county).strip().title() for county in county_scope if str(county).strip()}
            if county_scope
            else set(self.county_names)
        )
        missing = sorted(scoped_counties - covered)

        gap_rows: List[Dict[str, object]] = []
        covered_points = df[["latitude", "longitude"]].dropna()
        for county_name in missing:
            coords = self.county_centroids.get(county_name.lower())
            nearest_distance_km = None
            if coords and not covered_points.empty:
                dist = self._haversine_km(
                    lat1=float(coords[0]),
                    lon1=float(coords[1]),
                    lat2_series=covered_points["latitude"],
                    lon2_series=covered_points["longitude"],
                )
                if dist.notna().any():
                    nearest_distance_km = float(dist.min())
            gap_rows.append(
                {
                    "county": county_name,
                    "nearest_supplier_distance_km": nearest_distance_km,
                    "nearest_supplier_distance_miles": (
                        nearest_distance_km * 0.621371 if nearest_distance_km is not None else None
                    ),
                    "latitude": coords[0] if coords else None,
                    "longitude": coords[1] if coords else None,
                }
            )

        gap_df = pd.DataFrame(gap_rows)
        if not gap_df.empty:
            gap_df = gap_df.sort_values(
                ["nearest_supplier_distance_km", "county"],
                ascending=[False, True],
                na_position="last",
            )

        coverage_by_county = (
            df.groupby("county")
            .size()
            .reset_index(name="facility_count")
            .sort_values(["facility_count", "county"], ascending=[False, True])
        )
        return {
            "covered_counties": sorted(covered),
            "covered_county_count": int(len(covered)),
            "gap_counties": gap_df.head(int(max_gap_counties)).to_dict(orient="records") if not gap_df.empty else [],
            "gap_county_count": int(len(missing)),
            "scope_county_count": int(len(scoped_counties)),
            "coverage_by_county": coverage_by_county.head(25).to_dict(orient="records"),
        }
