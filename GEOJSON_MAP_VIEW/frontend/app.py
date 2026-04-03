from __future__ import annotations

import html
import json
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(
    page_title="Georgia EV Supply Chain GeoRAG",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BACKEND_CHAT_URL = os.getenv("BACKEND_CHAT_URL", "http://127.0.0.1:8000/chat")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
COUNTY_GEOJSON_PATH = PROJECT_ROOT / "data" / "Counties_Georgia.geojson"
CHAT_TIMEOUT_SECONDS = int(os.getenv("BACKEND_CHAT_TIMEOUT", "120"))
BACKEND_STARTUP_TIMEOUT_SECONDS = int(os.getenv("BACKEND_STARTUP_TIMEOUT", "35"))
AUTO_BACKEND_PORTS = [8000, 8001, 8002]
PALETTE = {
    "bg": "#f4f7f9",
    "panel": "#ffffff",
    "ink": "#11263a",
    "muted": "#5d7187",
    "border": "#d7e1ea",
    "teal": "#12897f",
    "blue": "#386fa4",
    "amber": "#d8902f",
    "red": "#b94b5c",
    "slate": "#586471",
}
SOURCE_LABELS = {
    "coordinates_excel": "Coordinate Workbook",
    "source_excel": "Source Excel",
    "county_centroid": "GeoJSON County Centroid",
    "missing": "Missing",
    "unknown": "Unknown",
}


@st.cache_data(show_spinner=False)
def load_county_geojson() -> dict | None:
    if not COUNTY_GEOJSON_PATH.exists():
        return None
    try:
        return json.loads(COUNTY_GEOJSON_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def health_url_from_chat_url(chat_url: str) -> str:
    if chat_url.endswith("/chat"):
        return chat_url[: -len("/chat")] + "/health"
    return chat_url.rstrip("/") + "/health"


def backend_is_healthy(chat_url: str) -> bool:
    health_url = health_url_from_chat_url(chat_url)
    try:
        with urllib.request.urlopen(health_url, timeout=3) as resp:
            if resp.status != 200:
                return False
            payload = json.loads(resp.read().decode("utf-8"))
            return bool(payload.get("pipeline_loaded"))
    except Exception:
        return False


def chat_url_from_port(port: int) -> str:
    return f"http://127.0.0.1:{int(port)}/chat"


def parse_port_from_url(url: str) -> int | None:
    try:
        parsed = urllib.parse.urlparse(url)
        return parsed.port
    except Exception:
        return None


def is_port_listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=1):
            return True
    except Exception:
        return False


def _windows_creationflags() -> int:
    flags = 0
    if os.name == "nt":
        detached = getattr(subprocess, "DETACHED_PROCESS", 0)
        new_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        flags = detached | new_group
    return flags


def wait_for_backend(chat_url: str, timeout_seconds: int = BACKEND_STARTUP_TIMEOUT_SECONDS) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if backend_is_healthy(chat_url):
            return True
        time.sleep(1)
    return False


def start_backend_on_port(port: int) -> bool:
    target_url = chat_url_from_port(port)
    if backend_is_healthy(target_url):
        return True
    try:
        subprocess.Popen(
            ["python", "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(int(port))],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=_windows_creationflags(),
        )
    except Exception:
        return False
    return wait_for_backend(target_url, timeout_seconds=BACKEND_STARTUP_TIMEOUT_SECONDS)


def discover_backend_url() -> str:
    preferred = st.session_state.get("backend_url") or BACKEND_CHAT_URL
    candidates = [preferred, BACKEND_CHAT_URL] + [chat_url_from_port(p) for p in AUTO_BACKEND_PORTS]

    deduped = []
    seen = set()
    for url in candidates:
        if url and url not in seen:
            seen.add(url)
            deduped.append(url)

    for url in deduped:
        if backend_is_healthy(url):
            st.session_state["backend_url"] = url
            return url

    # Try booting backend on preferred/default ports.
    ports = []
    preferred_port = parse_port_from_url(preferred)
    if preferred_port:
        ports.append(preferred_port)
    for p in AUTO_BACKEND_PORTS:
        if p not in ports:
            ports.append(p)

    # Attempt at most two start tries to avoid long UI blocking.
    attempts = 0
    for port in ports:
        if attempts >= 2:
            break
        url = chat_url_from_port(port)
        if is_port_listening(port) and not backend_is_healthy(url):
            continue
        attempts += 1
        if start_backend_on_port(port):
            st.session_state["backend_url"] = url
            return url

    raise urllib.error.URLError(
        "Backend is not reachable. Start it manually with: uvicorn backend.main:app --reload --port 8000"
    )


def call_backend(question: str) -> Dict:
    backend_url = discover_backend_url()

    payload = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(
        backend_url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=CHAT_TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read().decode("utf-8"))


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        #MainMenu, footer {{
            visibility: hidden;
        }}

        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(18, 137, 127, 0.10), transparent 30%),
                linear-gradient(180deg, #f8fbfc 0%, {PALETTE["bg"]} 42%, #eef3f6 100%);
            color: {PALETTE["ink"]};
        }}

        [data-testid="stHeader"] {{
            background: rgba(248, 251, 252, 0.82);
        }}

        .block-container {{
            max-width: 1720px;
            padding-top: 1.25rem;
            padding-bottom: 1.6rem;
        }}

        .hero-shell {{
            background: linear-gradient(135deg, #ffffff 0%, #f6fafb 58%, #eef7f7 100%);
            border: 1px solid {PALETTE["border"]};
            border-radius: 26px;
            padding: 1.55rem 1.7rem 1.45rem;
            box-shadow: 0 16px 36px rgba(15, 23, 42, 0.08);
        }}

        .hero-topline {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1rem;
        }}

        .hero-kicker {{
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: {PALETTE["teal"]};
            margin-bottom: 0.55rem;
        }}

        .hero-title {{
            font-size: clamp(2rem, 3.4vw, 3rem);
            line-height: 0.98;
            font-weight: 800;
            color: {PALETTE["ink"]};
            margin-bottom: 0.7rem;
            letter-spacing: -0.04em;
        }}

        .hero-subtitle {{
            font-size: 0.98rem;
            line-height: 1.55;
            color: {PALETTE["muted"]};
            max-width: 920px;
        }}

        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.55rem 0.85rem;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            border: 1px solid {PALETTE["border"]};
            white-space: nowrap;
        }}

        .status-online {{
            background: rgba(18, 137, 127, 0.08);
            color: {PALETTE["teal"]};
        }}

        .status-standby {{
            background: rgba(216, 144, 47, 0.10);
            color: {PALETTE["amber"]};
        }}

        .metrics-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 1rem;
        }}

        .metric-chip {{
            background: #f7fafc;
            border: 1px solid {PALETTE["border"]};
            border-radius: 18px;
            padding: 0.7rem 0.95rem;
            min-width: 150px;
        }}

        .metric-label {{
            font-size: 0.69rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: {PALETTE["muted"]};
            margin-bottom: 0.3rem;
        }}

        .metric-value {{
            font-size: 1rem;
            font-weight: 800;
            color: {PALETTE["ink"]};
            letter-spacing: -0.03em;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] > div {{
            border-radius: 24px !important;
            border: 1px solid {PALETTE["border"]} !important;
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.07);
            background: {PALETTE["panel"]};
        }}

        div[data-testid="stTextInput"] input {{
            background: #f8fafc;
            border: 1px solid {PALETTE["border"]};
            border-radius: 16px;
            color: {PALETTE["ink"]};
            padding: 0.95rem 1rem;
            font-size: 0.98rem;
        }}

        .stButton > button {{
            width: 100%;
            border: 0;
            border-radius: 16px;
            background: linear-gradient(135deg, {PALETTE["teal"]} 0%, #0f766e 100%);
            color: #ffffff;
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 0.92rem 1.15rem;
            box-shadow: 0 12px 24px rgba(15, 118, 110, 0.20);
        }}

        .stButton > button:hover {{
            background: linear-gradient(135deg, #0f766e 0%, #0b5f59 100%);
            color: #ffffff;
        }}

        h1, h2, h3, p, label, span, div {{
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}

        .section-eyebrow {{
            font-size: 0.71rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: {PALETTE["teal"]};
            margin-bottom: 0.35rem;
        }}

        .section-heading {{
            font-size: 1.2rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            color: {PALETTE["ink"]};
            margin-bottom: 0.2rem;
        }}

        .section-subtitle {{
            font-size: 0.82rem;
            line-height: 1.45;
            color: {PALETTE["muted"]};
            margin-bottom: 0.75rem;
        }}

        .answer-body {{
            font-size: 0.98rem;
            line-height: 1.65;
            color: {PALETTE["ink"]};
        }}

        .citation-chip {{
            display: inline-block;
            background: rgba(18, 137, 127, 0.10);
            color: {PALETTE["teal"]};
            border: 1px solid rgba(18, 137, 127, 0.16);
            border-radius: 999px;
            padding: 0.08rem 0.45rem;
            font-size: 0.78rem;
            font-weight: 800;
        }}

        .evidence-card {{
            background: #f8fbfc;
            border: 1px solid {PALETTE["border"]};
            border-radius: 18px;
            padding: 0.85rem 0.95rem;
            margin-bottom: 0.7rem;
        }}

        .evidence-topline {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.8rem;
            margin-bottom: 0.45rem;
        }}

        .chunk-pill {{
            display: inline-flex;
            align-items: center;
            background: rgba(56, 111, 164, 0.09);
            color: {PALETTE["blue"]};
            border-radius: 999px;
            padding: 0.22rem 0.55rem;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.08em;
        }}

        .engine-pill {{
            display: inline-flex;
            align-items: center;
            background: #eff4f8;
            color: {PALETTE["slate"]};
            border-radius: 999px;
            padding: 0.22rem 0.55rem;
            font-size: 0.69rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}

        .chunk-score {{
            font-size: 0.74rem;
            font-weight: 800;
            color: {PALETTE["teal"]};
        }}

        .chunk-company {{
            font-size: 0.9rem;
            font-weight: 800;
            color: {PALETTE["ink"]};
            margin-bottom: 0.28rem;
        }}

        .chunk-text {{
            font-size: 0.84rem;
            line-height: 1.5;
            color: {PALETTE["muted"]};
        }}

        .legend-wrap {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-bottom: 0.8rem;
        }}

        .legend-item {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.42rem 0.7rem;
            background: #f8fafc;
            border: 1px solid {PALETTE["border"]};
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 700;
            color: {PALETTE["muted"]};
        }}

        .legend-dot {{
            width: 0.7rem;
            height: 0.7rem;
            border-radius: 999px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_heading(eyebrow: str, title: str, subtitle: str = "") -> None:
    subtitle_html = f"<div class='section-subtitle'>{html.escape(subtitle)}</div>" if subtitle else ""
    st.markdown(
        f"""
        <div class="section-eyebrow">{html.escape(eyebrow)}</div>
        <div class="section-heading">{html.escape(title)}</div>
        {subtitle_html}
        """,
        unsafe_allow_html=True,
    )


def normalize_coordinate_source(source: object) -> str:
    if pd.isna(source):
        return SOURCE_LABELS["unknown"]
    text = str(source or "unknown").strip()
    lower = text.lower()
    if lower.startswith("coordinates_excel"):
        return SOURCE_LABELS["coordinates_excel"]
    return SOURCE_LABELS.get(lower, text or SOURCE_LABELS["unknown"])


def highlight_citations(answer: str) -> str:
    safe_text = html.escape(answer or "No answer returned.")
    safe_text = reformat_answer_breaks(safe_text)
    return safe_text


def reformat_answer_breaks(safe_text: str) -> str:
    def _replace_citation(match: object) -> str:
        citation = match.group(0)
        return f"<span class='citation-chip'>{citation}</span>"

    text = safe_text.replace("\n", "<br>")
    return re.sub(r"\[C\d+\]", _replace_citation, text)


def compact_text(value: object, limit: int = 280) -> str:
    if pd.isna(value):
        return ""
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def render_header(backend_ok: bool) -> None:
    status_label = "Pipeline Online" if backend_ok else "Pipeline Standby"
    status_class = "status-online" if backend_ok else "status-standby"
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-topline">
                <div>
                    <div class="hero-kicker">Hybrid GeoJSON + Geospatial RAG Platform</div>
                    <div class="hero-title">Georgia EV Supply Chain Intelligence Dashboard</div>
                    <div class="hero-subtitle">
                        Structured SQL retrieval, FAISS semantic search, GeoJSON county mapping,
                        and local evidence-grounded LLM synthesis in one research-ready interface.
                    </div>
                </div>
                <div class="status-pill {status_class}">{status_label}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_query_panel() -> tuple[bool, str]:
    section_heading(
        "Query",
        "Research Question",
        "Submit one geospatial supply-chain question and inspect grounded evidence, ranked companies, and county-aware map output.",
    )
    with st.form("research_query_form", clear_on_submit=False, border=False):
        input_col, button_col = st.columns([0.82, 0.18], vertical_alignment="bottom")
        with input_col:
            question = st.text_input(
                "Research question",
                key="research_query_text",
                placeholder="",
                label_visibility="collapsed",
            )
        with button_col:
            submitted = st.form_submit_button("Run Query", use_container_width=True)
    return submitted, question.strip()


def render_result_metrics(result: Dict) -> None:
    plan = result.get("plan", {}) if isinstance(result.get("plan", {}), dict) else {}
    chunks = result.get("retrieved_chunks", []) or []
    companies = result.get("retrieved_companies", []) or []
    metric_values = [
        ("Query Plan", str(plan.get("classification", "N/A")).replace("_", " ")),
        ("Evidence Chunks", f"{len(chunks)}"),
        ("Mapped Companies", f"{len(companies)}"),
        ("LLM Model", str(result.get("model_used", "unknown"))),
    ]
    metric_html = "".join(
        f"""
        <div class="metric-chip">
            <div class="metric-label">{html.escape(label)}</div>
            <div class="metric-value">{html.escape(value)}</div>
        </div>
        """
        for label, value in metric_values
    )
    st.markdown(f"<div class='metrics-row'>{metric_html}</div>", unsafe_allow_html=True)


def render_answer_panel(question: str, answer: str) -> None:
    with st.container(border=True):
        section_heading(
            "Generated Answer",
            "Evidence-Grounded Synthesis",
            f"Question: {question}" if question else "",
        )
        st.markdown(
            f"<div class='answer-body'>{highlight_citations(answer)}</div>",
            unsafe_allow_html=True,
        )


def render_chunks(chunks: List[Dict]) -> None:
    with st.container(border=True):
        section_heading(
            "Retrieved Evidence",
            "Supporting Chunks",
            "Chunk IDs, retrieval engine, company context, and relevance scores used to generate the final answer.",
        )
        if not chunks:
            st.info("No supporting chunks were returned for this query.")
            return

        for chunk in chunks[:5]:
            chunk_id = html.escape(str(chunk.get("chunk_id") or "C?"))
            engine = html.escape(str(chunk.get("engine") or "unknown"))
            chunk_type = html.escape(str(chunk.get("chunk_type") or "retrieved_chunk").replace("_", " "))
            company = html.escape(str(chunk.get("company") or "Dataset-level evidence"))
            score = chunk.get("score", 0.0)
            try:
                score_text = f"{float(score):.3f}"
            except Exception:
                score_text = str(score)
            chunk_text = html.escape(compact_text(chunk.get("text", ""), limit=260) or "No text payload.")
            st.markdown(
                f"""
                <div class="evidence-card">
                    <div class="evidence-topline">
                        <div>
                            <span class="chunk-pill">{chunk_id}</span>
                            <span class="engine-pill">{engine}</span>
                            <span class="engine-pill">{chunk_type}</span>
                        </div>
                        <div class="chunk-score">score {html.escape(score_text)}</div>
                    </div>
                    <div class="chunk-company">{company}</div>
                    <div class="chunk-text">{chunk_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_table(records: List[Dict]) -> None:
    with st.container(border=True):
        section_heading(
            "Structured Results",
            "Retrieved Companies",
            "Ranked company records with location, OEM, distance, map weight, and coordinate provenance.",
        )
        if not records:
            st.info("No company records were returned for this query.")
            return

        df = pd.DataFrame(records).copy()
        if "distance_km" in df.columns:
            df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce").round(2)
        if "map_weight" in df.columns:
            df["map_weight"] = pd.to_numeric(df["map_weight"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        if "employment" in df.columns:
            df["employment"] = pd.to_numeric(df["employment"], errors="coerce")
        if "coordinate_source" in df.columns:
            df["coordinate_source"] = df["coordinate_source"].apply(normalize_coordinate_source)

        if "distance_km" in df.columns and df["distance_km"].notna().any():
            secondary_sort = "map_weight" if "map_weight" in df.columns else ("company" if "company" in df.columns else None)
            sort_cols = ["distance_km"] + ([secondary_sort] if secondary_sort else [])
            ascending = [True] + ([False] if secondary_sort == "map_weight" else [True] if secondary_sort else [])
            df = df.sort_values(sort_cols, ascending=ascending)
        elif "map_weight" in df.columns:
            df = df.sort_values("map_weight", ascending=False)

        rename_map = {
            "company": "Company",
            "industry_group": "Industry",
            "city": "City",
            "county": "County",
            "primary_oems": "Primary OEMs",
            "ev_supply_chain_role": "EV Role",
            "employment": "Employment",
            "distance_km": "Distance (km)",
            "coordinate_source": "Coordinate Source",
            "map_weight": "Map Weight",
            "score": "Score",
        }
        preferred_cols = [
            "company",
            "industry_group",
            "ev_supply_chain_role",
            "city",
            "county",
            "primary_oems",
            "distance_km",
            "map_weight",
            "employment",
            "coordinate_source",
        ]
        ordered = [c for c in preferred_cols if c in df.columns]
        display_df = df[ordered].rename(columns=rename_map).head(18)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=390,
            column_config={
                "Map Weight": st.column_config.ProgressColumn("Map Weight", min_value=0.0, max_value=1.0, format="%.2f"),
                "Distance (km)": st.column_config.NumberColumn("Distance (km)", format="%.2f"),
                "Employment": st.column_config.NumberColumn("Employment", format="%d"),
            },
        )


def render_map(records: List[Dict]) -> None:
    with st.container(border=True):
        section_heading(
            "Geospatial Map",
            "GeoJSON County Map + Supplier Heat Layer",
            "County boundaries, weighted company markers, and coordinate-source legend for screenshot-ready spatial interpretation.",
        )

        if not records:
            st.info("No mapped companies were returned for this query.")
            return

        df = pd.DataFrame(records).copy()
        if "latitude" not in df.columns or "longitude" not in df.columns:
            st.info("Returned company rows do not contain latitude/longitude columns.")
            return

        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df = df.dropna(subset=["latitude", "longitude"]).copy()
        if df.empty:
            st.info("No valid coordinates are available for the retrieved companies.")
            return

        if "map_weight" not in df.columns:
            score_fallback = pd.to_numeric(df.get("score"), errors="coerce").fillna(0.5)
            df["map_weight"] = score_fallback.clip(lower=0.05, upper=1.0)
        else:
            df["map_weight"] = pd.to_numeric(df["map_weight"], errors="coerce").fillna(0.5).clip(lower=0.05, upper=1.0)

        df["radius"] = df["map_weight"].apply(lambda v: 2800.0 + float(v) * 22000.0)
        df["heat_weight"] = df["map_weight"].apply(lambda v: 8.0 + float(v) * 88.0)
        if "coordinate_source" not in df.columns:
            df["coordinate_source"] = "unknown"
        df["tooltip_company"] = df.get("company", pd.Series(index=df.index)).fillna("Unknown company")
        df["tooltip_role"] = df.get("ev_supply_chain_role", pd.Series(index=df.index)).fillna("Unknown role")
        df["tooltip_location"] = (
            df.get("city", pd.Series(index=df.index)).fillna("")
            + ", "
            + df.get("county", pd.Series(index=df.index)).fillna("")
        ).str.strip(", ")
        df["tooltip_coord_source"] = df.get("coordinate_source", pd.Series(index=df.index)).apply(normalize_coordinate_source)
        df["tooltip_weight"] = pd.to_numeric(df["map_weight"], errors="coerce").fillna(0.0).map(lambda v: f"{float(v):.2f}")

        color_lookup = {
            "coordinates_excel": [18, 137, 127, 220],
            "source_excel": [56, 111, 164, 220],
            "county_centroid": [216, 144, 47, 215],
            "missing": [185, 75, 92, 190],
            "unknown": [88, 100, 113, 185],
        }

        def _point_color(source: object) -> List[int]:
            text = str(source or "").lower()
            if text.startswith("coordinates_excel"):
                return color_lookup["coordinates_excel"]
            return color_lookup.get(text, color_lookup["unknown"])

        df["fill_color"] = df["coordinate_source"].apply(_point_color)

        legend_html = """
            <div class="legend-wrap">
                <div class="legend-item"><span class="legend-dot" style="background:#12897f"></span> Coordinate Workbook</div>
                <div class="legend-item"><span class="legend-dot" style="background:#386fa4"></span> Source Excel</div>
                <div class="legend-item"><span class="legend-dot" style="background:#d8902f"></span> GeoJSON County Centroid</div>
                <div class="legend-item"><span class="legend-dot" style="background:#b94b5c"></span> Missing</div>
            </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)

        view_state = pdk.ViewState(
            latitude=float(df["latitude"].mean()),
            longitude=float(df["longitude"].mean()),
            zoom=6.4 if len(df) > 8 else 7.6,
            pitch=0,
        )

        heatmap_layer = pdk.Layer(
            "HeatmapLayer",
            data=df,
            get_position="[longitude, latitude]",
            get_weight="heat_weight",
            radius_pixels=42,
            intensity=0.82,
            threshold=0.06,
            opacity=0.42,
        )

        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position="[longitude, latitude]",
            get_fill_color="fill_color",
            get_line_color=[17, 38, 58, 190],
            line_width_min_pixels=1,
            stroked=True,
            filled=True,
            pickable=True,
            get_radius="radius",
        )

        county_geojson = load_county_geojson()
        county_layer = None
        if county_geojson:
            county_layer = pdk.Layer(
                "GeoJsonLayer",
                data=county_geojson,
                stroked=True,
                filled=False,
                get_line_color=[17, 38, 58, 150],
                line_width_min_pixels=2,
            )

        tooltip = {
            "html": (
                "<div style='font-family:Inter, sans-serif; min-width:220px;'>"
                "<div style='font-size:14px; font-weight:800; margin-bottom:8px;'>{tooltip_company}</div>"
                "<div style='font-size:12px; line-height:1.6; color:#e5eef3;'>"
                "<b>Role:</b> {tooltip_role}<br/>"
                "<b>Location:</b> {tooltip_location}<br/>"
                "<b>Coordinate:</b> {tooltip_coord_source}<br/>"
                "<b>Map weight:</b> {tooltip_weight}"
                "</div></div>"
            ),
            "style": {
                "backgroundColor": "#102433",
                "color": "#ffffff",
                "borderRadius": "16px",
                "padding": "14px 16px",
                "border": "1px solid rgba(255,255,255,0.10)",
            },
        }

        st.pydeck_chart(
            pdk.Deck(
                map_style="light_no_labels",
                initial_view_state=view_state,
                tooltip=tooltip,
                layers=[layer for layer in [county_layer, heatmap_layer, scatter_layer] if layer is not None],
            ),
            use_container_width=True,
        )


def initialize_state() -> None:
    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None
    if "latest_question" not in st.session_state:
        st.session_state.latest_question = ""
    if "latest_error" not in st.session_state:
        st.session_state.latest_error = None
    if "research_query_text" not in st.session_state:
        st.session_state.research_query_text = ""


def run_query(question: str) -> None:
    try:
        with st.spinner("Retrieving evidence from SQL, Geo, and Vector engines..."):
            st.session_state.latest_result = call_backend(question)
        st.session_state.latest_question = question
        st.session_state.latest_error = None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        st.session_state.latest_error = f"Backend HTTP error: {exc.code} - {detail}"
    except urllib.error.URLError as exc:
        st.session_state.latest_error = f"Backend connection failed: {exc.reason}"
    except Exception as exc:
        st.session_state.latest_error = f"Unexpected error: {exc}"


inject_styles()
initialize_state()

effective_url = st.session_state.get("backend_url", BACKEND_CHAT_URL)
backend_ok = backend_is_healthy(effective_url)
render_header(backend_ok)

st.write("")
submitted, user_question = render_query_panel()
if submitted:
    if user_question:
        run_query(user_question)
    else:
        st.session_state.latest_error = "Please enter a research question before running retrieval."

if st.session_state.latest_error:
    st.error(st.session_state.latest_error)

result = st.session_state.latest_result
if result:
    render_result_metrics(result)
    st.write("")

    answer_col, output_col = st.columns([0.96, 1.04], gap="large")
    with answer_col:
        render_answer_panel(
            question=st.session_state.latest_question,
            answer=result.get("answer", "No answer returned."),
        )
        st.write("")
        render_chunks(result.get("retrieved_chunks", []) or [])

    with output_col:
        render_map(result.get("retrieved_companies", []) or [])
        st.write("")
        render_table(result.get("retrieved_companies", []) or [])
