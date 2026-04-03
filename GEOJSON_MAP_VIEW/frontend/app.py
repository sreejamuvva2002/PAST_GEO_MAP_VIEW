from __future__ import annotations

import html
import json
import math
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
    page_title="Georgia EV Supply Chain GeoRAG Chatbot",
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

        [data-testid="stHeaderActionElements"],
        [data-testid="stHeadingActionElements"],
        [data-testid="stMarkdownContainer"] a[href^="#"],
        [data-testid="stMarkdownContainer"] .anchor-link,
        [data-testid="stMarkdownContainer"] [class*="anchor"],
        [data-testid="stMarkdownContainer"] [class*="Anchor"],
        [data-testid="stMarkdownContainer"] [aria-label*="link" i],
        .answer-body a[href^="#"],
        .answer-body h1 a,
        .answer-body h2 a,
        .answer-body h3 a {{
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            height: 0 !important;
        }}

        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(18, 137, 127, 0.10), transparent 30%),
                linear-gradient(180deg, #f8fbfc 0%, {PALETTE["bg"]} 42%, #eef3f6 100%);
            color: {PALETTE["ink"]};
            overflow-x: hidden;
        }}

        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > .main {{
            overflow-x: hidden;
        }}

        [data-testid="stMainBlockContainer"] {{
            width: calc(100% - clamp(1.5rem, 2.4vw, 2.5rem)) !important;
            max-width: calc(100vw - 8rem) !important;
            margin-right: clamp(1.5rem, 2.4vw, 2.5rem) !important;
            box-sizing: border-box !important;
            overflow-x: hidden;
        }}

        [data-testid="stHeader"] {{
            background: rgba(248, 251, 252, 0.82);
        }}

        .block-container {{
            width: calc(100% - clamp(1.5rem, 2.4vw, 2.5rem));
            max-width: min(1720px, calc(100vw - 8rem));
            margin-left: auto;
            margin-right: clamp(1.5rem, 2.4vw, 2.5rem);
            padding-top: 1.25rem;
            padding-left: clamp(1.5rem, 2.4vw, 2.5rem) !important;
            padding-right: clamp(1.5rem, 2.4vw, 2.5rem) !important;
            padding-bottom: 1.6rem;
            box-sizing: border-box;
        }}

        .hero-shell {{
            background:
                radial-gradient(circle at top right, rgba(18, 137, 127, 0.08), transparent 32%),
                linear-gradient(135deg, #ffffff 0%, #f7fafc 60%, #eef7f7 100%);
            border: 1px solid {PALETTE["border"]};
            border-radius: 28px;
            width: 100%;
            max-width: 100%;
            margin-top: 1.35rem;
            padding: 1.4rem 1.6rem 1.35rem;
            box-shadow: 0 18px 42px rgba(15, 23, 42, 0.08);
            box-sizing: border-box;
            transition: transform 180ms ease, box-shadow 180ms ease;
        }}

        .hero-shell:hover {{
            transform: translateY(-2px);
            box-shadow: 0 22px 48px rgba(15, 23, 42, 0.11);
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
            font-size: clamp(2rem, 3vw, 2.8rem);
            line-height: 1.02;
            font-weight: 800;
            color: {PALETTE["ink"]};
            margin-bottom: 0.65rem;
            letter-spacing: -0.04em;
        }}

        .hero-subtitle {{
            font-size: 0.96rem;
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
            display: none;
        }}

        [data-testid="stMetric"] {{
            background: linear-gradient(180deg, #ffffff 0%, #f8fbfc 100%);
            border: 1px solid #d7e1ea;
            border-radius: 22px;
            padding: 0.9rem 1rem;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06);
            transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
        }}

        [data-testid="stMetric"]:hover {{
            transform: translateY(-2px);
            border-color: rgba(18, 137, 127, 0.22);
            box-shadow: 0 18px 36px rgba(15, 23, 42, 0.09);
        }}

        [data-testid="stMetricLabel"] {{
            font-size: 0.69rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: {PALETTE["muted"]};
        }}

        [data-testid="stMetricValue"] {{
            font-size: 1rem;
            font-weight: 800;
            color: {PALETTE["ink"]};
            letter-spacing: -0.03em;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            width: 100%;
            max-width: 100%;
            box-sizing: border-box;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] > div {{
            border-radius: 24px !important;
            border: 1px solid {PALETTE["border"]} !important;
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.07);
            background: {PALETTE["panel"]};
            width: 100% !important;
            max-width: 100% !important;
            padding: 1.15rem 1.3rem 1.3rem !important;
            box-sizing: border-box !important;
            overflow: hidden;
            transition: box-shadow 160ms ease, border-color 160ms ease;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] > div:hover {{
            border-color: rgba(18, 137, 127, 0.16) !important;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.09);
        }}

        div[data-testid="stTextInput"] input {{
            background: #ffffff;
            border: 1px solid #c9d7e5;
            border-radius: 18px;
            color: {PALETTE["ink"]};
            padding: 1rem 1.05rem;
            font-size: 0.96rem;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.05);
            transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
        }}

        div[data-testid="stTextInput"] input:focus {{
            border-color: rgba(18, 137, 127, 0.55);
            box-shadow: 0 0 0 4px rgba(18, 137, 127, 0.10), 0 14px 28px rgba(15, 23, 42, 0.06);
        }}

        button[data-testid="stBaseButton-primary"] {{
            width: calc(100% - 0.35rem);
            max-width: calc(100% - 0.35rem);
            border: 0;
            border-radius: 18px;
            background: linear-gradient(135deg, #12897f 0%, #0f766e 45%, #0b5f59 100%);
            color: #ffffff;
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 0.98rem 1.15rem;
            margin-right: 0.35rem;
            box-sizing: border-box;
            box-shadow: 0 14px 28px rgba(15, 118, 110, 0.22);
            transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
        }}

        button[data-testid="stBaseButton-primary"]:hover {{
            background: linear-gradient(135deg, #0f766e 0%, #0b5f59 100%);
            color: #ffffff;
            transform: translateY(-1px);
            box-shadow: 0 18px 32px rgba(15, 118, 110, 0.28);
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
            margin-bottom: 0.4rem;
            text-align: left;
        }}

        .section-heading {{
            font-size: 1.22rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            color: {PALETTE["ink"]};
            margin-bottom: 0.28rem;
            text-align: left;
        }}

        .section-subtitle {{
            font-size: 0.82rem;
            line-height: 1.55;
            color: {PALETTE["muted"]};
            margin-bottom: 0.75rem;
            text-align: left;
        }}

        .answer-body {{
            font-size: 0.95rem;
            line-height: 1.78;
            color: {PALETTE["ink"]};
            letter-spacing: -0.015em;
            text-align: left;
            padding: 0 0.1rem 0.15rem;
            max-width: 100%;
            overflow-wrap: anywhere;
            word-break: break-word;
        }}

        .citation-chip {{
            display: inline-block;
            background: rgba(18, 137, 127, 0.10);
            color: {PALETTE["teal"]};
            border: 1px solid rgba(18, 137, 127, 0.16);
            border-radius: 999px;
            padding: 0.1rem 0.48rem;
            font-size: 0.78rem;
            font-weight: 800;
            margin-left: 0.12rem;
        }}

        .evidence-card {{
            background: linear-gradient(180deg, #ffffff 0%, #f8fbfc 100%);
            border: 1px solid {PALETTE["border"]};
            border-radius: 20px;
            width: 100%;
            max-width: 100%;
            padding: 0.9rem 1rem;
            margin-bottom: 0.7rem;
            box-sizing: border-box;
            transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
        }}

        .evidence-card:hover {{
            transform: translateY(-2px);
            border-color: rgba(56, 111, 164, 0.22);
            box-shadow: 0 14px 28px rgba(15, 23, 42, 0.06);
        }}

        .evidence-topline {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.8rem;
            margin-bottom: 0.45rem;
            width: 100%;
            box-sizing: border-box;
            padding-right: 0.2rem;
        }}

        .evidence-topline > div:first-child {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.45rem;
            min-width: 0;
            flex: 1 1 auto;
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
            flex: 0 0 auto;
            min-width: 6.8rem;
            padding-right: 0.25rem;
            font-size: 0.74rem;
            font-weight: 800;
            color: {PALETTE["teal"]};
            text-align: right;
            white-space: nowrap;
        }}

        .chunk-company {{
            font-size: 0.94rem;
            font-weight: 800;
            color: {PALETTE["ink"]};
            margin-bottom: 0.32rem;
            text-align: left;
            max-width: 100%;
            overflow-wrap: anywhere;
            word-break: break-word;
        }}

        .chunk-text {{
            font-size: 0.86rem;
            line-height: 1.6;
            color: {PALETTE["muted"]};
            text-align: left;
            max-width: 100%;
            overflow-wrap: anywhere;
            word-break: break-word;
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

        .query-shell {{
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid {PALETTE["border"]};
            border-radius: 24px;
            width: 100%;
            max-width: 100%;
            padding: 1rem 1rem 0.3rem;
            box-shadow: 0 16px 32px rgba(15, 23, 42, 0.07);
            box-sizing: border-box;
            margin-bottom: 0.5rem;
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
        }}

        .query-shell:hover {{
            transform: translateY(-1px);
            border-color: rgba(18, 137, 127, 0.18);
            box-shadow: 0 20px 38px rgba(15, 23, 42, 0.09);
        }}

        .context-card {{
            background: linear-gradient(180deg, #ffffff 0%, #f8fbfc 100%);
            border: 1px solid {PALETTE["border"]};
            border-radius: 20px;
            width: 100%;
            max-width: 100%;
            padding: 0.85rem 1rem;
            margin-bottom: 0.7rem;
            box-sizing: border-box;
            transition: transform 160ms ease, box-shadow 160ms ease;
        }}

        .context-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 12px 26px rgba(15, 23, 42, 0.06);
        }}

        .context-label {{
            font-size: 0.68rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: {PALETTE["muted"]};
            margin-bottom: 0.3rem;
            text-align: left;
        }}

        .context-value {{
            font-size: 0.94rem;
            font-weight: 800;
            color: {PALETTE["ink"]};
            letter-spacing: -0.03em;
            text-align: left;
        }}

        .context-note {{
            margin-top: 0.35rem;
            font-size: 0.74rem;
            line-height: 1.45;
            color: {PALETTE["muted"]};
            text-align: left;
        }}

        .answer-body h3 {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            margin: 1rem 0 0.55rem;
            padding: 0.42rem 0.72rem;
            border-radius: 999px;
            background: rgba(18, 137, 127, 0.08);
            border: 1px solid rgba(18, 137, 127, 0.14);
            font-size: 0.82rem;
            font-weight: 800;
            color: {PALETTE["teal"]};
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }}

        .answer-body h3::before {{
            content: "";
            display: inline-block;
            width: 0.48rem;
            height: 0.48rem;
            border-radius: 999px;
            background: linear-gradient(135deg, #12897f, #0f766e);
        }}

        .answer-body ul {{
            margin-top: 0.45rem;
            padding-left: 1.1rem;
        }}

        .answer-body li {{
            margin-bottom: 0.3rem;
        }}

        div[data-testid="stDataFrame"] {{
            border-radius: 18px;
            overflow: hidden;
        }}

        div[data-testid="stDataFrame"] div[role="grid"] {{
            border: 1px solid #e3e8ef;
            border-radius: 18px;
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
    cleaned = clean_generated_answer(answer)
    safe_text = html.escape(cleaned or "No answer returned.")
    safe_text = reformat_answer_breaks(safe_text)
    return safe_text


def clean_generated_answer(answer: str) -> str:
    text = str(answer or "").strip()
    text = re.sub(r"(?im)^\s*Question\s*:\s*.*(?:\n|$)", "", text)
    text = re.sub(r"(?m)^\s*#{1,6}\s*$", "", text)
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def reformat_answer_breaks(safe_text: str) -> str:
    def _replace_citation(match: object) -> str:
        citation = match.group(0)
        return f"<span class='citation-chip'>{citation}</span>"

    text = safe_text.replace("\n", "<br>")
    for heading in ["Direct Answer", "Spatial / Supply-Chain Details", "Evidence Gaps"]:
        text = text.replace(f"{heading}<br>", f"<h3>{heading}</h3>")
        if text.endswith(heading):
            text = text[: -len(heading)] + f"<h3>{heading}</h3>"
    return re.sub(r"\[C\d+\]", _replace_citation, text)


def compact_text(value: object, limit: int = 280) -> str:
    if pd.isna(value):
        return ""
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def render_header(backend_ok: bool) -> None:
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-topline">
                <div>
                    <div class="hero-kicker">Hybrid GeoJSON + Geospatial RAG</div>
                    <div class="hero-title">Georgia EV Supply Chain Intelligence Chatbot</div>
                    <div class="hero-subtitle">
                        County-aware GeoJSON mapping, radius and disruption analysis, hybrid SQL + FAISS retrieval,
                        and citation-grounded LLM synthesis for Georgia EV supply-chain intelligence.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_query_panel() -> tuple[bool, str]:
    with st.container():
        st.markdown("<div class='query-shell'>", unsafe_allow_html=True)
        section_heading(
            "Query",
            "Ask a Geospatial Supply-Chain Question",
            "Search by county, radius, supplier capability, disruption scenario, or network gaps across Georgia.",
        )
        with st.form("research_query_form", clear_on_submit=False, border=False):
            input_col, button_col = st.columns([0.84, 0.16], vertical_alignment="bottom")
            with input_col:
                question = st.text_input(
                    "Research question",
                    key="research_query_text",
                    placeholder="Example: Show battery suppliers within 20 miles of 33.7490, -84.3880.",
                    label_visibility="collapsed",
                )
            with button_col:
                submitted = st.form_submit_button("Run GeoRAG", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)
    return submitted, question.strip()


def render_result_metrics(result: Dict) -> None:
    map_context = result.get("map_context", {}) if isinstance(result.get("map_context", {}), dict) else {}
    chunks = result.get("retrieved_chunks", []) or []
    companies = result.get("retrieved_companies", []) or []
    gap_report = map_context.get("gap_report", {}) if isinstance(map_context.get("gap_report", {}), dict) else {}
    radius_km = map_context.get("radius_km")
    if radius_km is None:
        radius_label = "County / semantic"
    else:
        radius_label = f"{float(radius_km):.1f} km / {float(radius_km) * 0.621371:.1f} mi"
    metric_values = [
        ("Evidence Chunks", f"{len(chunks)}"),
        ("Mapped Companies", f"{len(companies)}"),
        ("County Coverage", f"{int(map_context.get('county_coverage_count') or 0)}"),
        ("Gap Counties", f"{int(gap_report.get('gap_county_count') or 0)}"),
        ("Search Radius", radius_label),
        ("LLM Model", str(result.get("model_used", "unknown"))),
    ]
    for start in range(0, len(metric_values), 4):
        row_items = metric_values[start : start + 4]
        metric_cols = st.columns(4)
        for idx, col in enumerate(metric_cols):
            if idx < len(row_items):
                label, value = row_items[idx]
                col.metric(label, value)
            else:
                col.empty()


def render_map_context_summary(result: Dict) -> None:
    map_context = result.get("map_context", {}) if isinstance(result.get("map_context", {}), dict) else {}
    if not map_context:
        return

    gap_report = map_context.get("gap_report", {}) if isinstance(map_context.get("gap_report", {}), dict) else {}
    focus_label = map_context.get("focus_label") or "Dataset scope"
    map_mode = str(map_context.get("map_mode") or "standard").replace("_", " ").title()
    center_lat = map_context.get("center_lat")
    center_lon = map_context.get("center_lon")
    counties = ", ".join(map_context.get("counties", []) or []) or "All Georgia counties"
    if center_lat is not None and center_lon is not None:
        center_note = f"Search hub at {float(center_lat):.4f}, {float(center_lon):.4f}"
    else:
        center_note = "No explicit search hub; map displays retrieved facility coordinates and county coverage."
    if map_context.get("radius_km") is not None:
        radius_note = f"Radius {float(map_context['radius_km']):.1f} km / {float(map_context['radius_km']) * 0.621371:.1f} mi"
    else:
        radius_note = "County and semantic retrieval mode"

    cards = [
        ("Map Mode", map_mode, radius_note),
        ("Focus", str(focus_label), center_note),
        ("County Scope", counties, f"Covered counties in result: {int(map_context.get('county_coverage_count') or 0)}"),
        (
            "Gap Scan",
            f"{int(gap_report.get('gap_county_count') or 0)} counties",
            "Highest-distance uncovered counties are highlighted in red when gap analysis is active.",
        ),
    ]
    card_html = "".join(
        f"""
        <div class="context-card">
            <div class="context-label">{html.escape(label)}</div>
            <div class="context-value">{html.escape(value)}</div>
            <div class="context-note">{html.escape(note)}</div>
        </div>
        """
        for label, value, note in cards
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_answer_panel(question: str, answer: str) -> None:
    with st.container(border=True):
        section_heading(
            "Generated Answer",
            "Evidence-Grounded Synthesis",
            "",
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

        for chunk in chunks[:8]:
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
        if "distance_miles" in df.columns:
            df["distance_miles"] = pd.to_numeric(df["distance_miles"], errors="coerce").round(2)
        if "nearest_peer_distance_km" in df.columns:
            df["nearest_peer_distance_km"] = pd.to_numeric(df["nearest_peer_distance_km"], errors="coerce").round(2)
        if "nearest_peer_distance_miles" in df.columns:
            df["nearest_peer_distance_miles"] = pd.to_numeric(df["nearest_peer_distance_miles"], errors="coerce").round(2)
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
            "category": "Category",
            "industry_group": "Industry",
            "city": "City",
            "county": "County",
            "address": "Address",
            "primary_oems": "Primary OEMs",
            "ev_supply_chain_role": "EV Role",
            "product_service": "Product / Service",
            "ev_battery_relevant": "EV / Battery",
            "employment": "Employment",
            "distance_km": "Distance (km)",
            "distance_miles": "Distance (mi)",
            "nearest_peer_company": "Nearest Peer",
            "nearest_peer_distance_km": "Nearest Peer (km)",
            "nearest_peer_distance_miles": "Nearest Peer (mi)",
            "match_reason": "Match Reason",
            "coordinate_source": "Coordinate Source",
            "map_weight": "Map Weight",
            "score": "Score",
        }
        preferred_cols = [
            "company",
            "category",
            "industry_group",
            "ev_supply_chain_role",
            "product_service",
            "city",
            "county",
            "address",
            "primary_oems",
            "distance_km",
            "distance_miles",
            "nearest_peer_company",
            "nearest_peer_distance_km",
            "nearest_peer_distance_miles",
            "map_weight",
            "match_reason",
            "ev_battery_relevant",
            "employment",
            "coordinate_source",
        ]
        ordered = [c for c in preferred_cols if c in df.columns]
        display_df = df[ordered].rename(columns=rename_map).head(18)
        table_height = min(560, max(128, 52 + (len(display_df) + 1) * 42))

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=table_height,
            column_config={
                "Map Weight": st.column_config.ProgressColumn("Map Weight", min_value=0.0, max_value=1.0, format="%.2f"),
                "Distance (km)": st.column_config.NumberColumn("Distance (km)", format="%.2f"),
                "Distance (mi)": st.column_config.NumberColumn("Distance (mi)", format="%.2f"),
                "Nearest Peer (km)": st.column_config.NumberColumn("Nearest Peer (km)", format="%.2f"),
                "Nearest Peer (mi)": st.column_config.NumberColumn("Nearest Peer (mi)", format="%.2f"),
                "Employment": st.column_config.NumberColumn("Employment", format="%d"),
            },
        )


def build_radius_circle_geojson(center_lat: float, center_lon: float, radius_km: float, steps: int = 96) -> dict:
    earth_radius_km = 6371.0088
    lat_rad = math.radians(center_lat)
    lon_rad = math.radians(center_lon)
    angular_distance = float(radius_km) / earth_radius_km
    coords = []
    for step in range(steps + 1):
        bearing = 2.0 * math.pi * step / steps
        point_lat = math.asin(
            math.sin(lat_rad) * math.cos(angular_distance)
            + math.cos(lat_rad) * math.sin(angular_distance) * math.cos(bearing)
        )
        point_lon = lon_rad + math.atan2(
            math.sin(bearing) * math.sin(angular_distance) * math.cos(lat_rad),
            math.cos(angular_distance) - math.sin(lat_rad) * math.sin(point_lat),
        )
        coords.append([math.degrees(point_lon), math.degrees(point_lat)])
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"fill_color": [18, 137, 127, 26]},
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            }
        ],
    }


def build_county_overlay_geojson(
    df: pd.DataFrame,
    map_context: Dict,
) -> dict | None:
    county_geojson = load_county_geojson()
    if not county_geojson:
        return None

    counts = {}
    if "county" in df.columns:
        counts = (
            df["county"].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().value_counts().to_dict()
        )

    gap_report = map_context.get("gap_report", {}) if isinstance(map_context.get("gap_report", {}), dict) else {}
    gap_names = {
        str(item.get("county") or "").strip().lower()
        for item in gap_report.get("gap_counties", []) or []
        if str(item.get("county") or "").strip()
    }
    selected = {
        str(county or "").strip().lower().replace(" county", "")
        for county in map_context.get("counties", []) or []
        if str(county or "").strip()
    }

    max_count = max(counts.values()) if counts else 1
    overlay = json.loads(json.dumps(county_geojson))
    for feature in overlay.get("features", []):
        props = feature.setdefault("properties", {})
        county_name = str(props.get("NAME10") or "").strip()
        county_key = county_name.lower()
        count = int(counts.get(county_name, 0))
        intensity = min(180, 35 + int(180 * count / max_count)) if count else 18

        if county_key in gap_names:
            fill = [185, 75, 92, 120]
        elif county_key in selected:
            fill = [18, 137, 127, 95]
        elif count:
            fill = [56, 111, 164, intensity]
        else:
            fill = [110, 126, 142, 10]

        props["fill_color"] = fill
        props["line_color"] = [17, 38, 58, 130]
        props["facility_count"] = count
    return overlay


def build_center_and_arc_frames(df: pd.DataFrame, map_context: Dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    center_lat = map_context.get("center_lat")
    center_lon = map_context.get("center_lon")
    if center_lat is None or center_lon is None:
        return pd.DataFrame(), pd.DataFrame()

    center_df = pd.DataFrame(
        [
            {
                "latitude": float(center_lat),
                "longitude": float(center_lon),
                "label": str(map_context.get("focus_label") or "Search center"),
                "radius": 7000,
                "tooltip_company": str(map_context.get("focus_label") or "Search center"),
                "tooltip_role": "Search center",
                "tooltip_product": "Query focus",
                "tooltip_location": str(map_context.get("focus_label") or "Search center"),
                "tooltip_coord_source": "Query center",
                "tooltip_weight": "1.00",
                "tooltip_distance": "0.00 km / 0.00 mi",
                "tooltip_nearest_peer": "N/A",
                "tooltip_nearest_peer_distance": "N/A",
            }
        ]
    )
    arc_df = df.copy()
    arc_df["source_latitude"] = float(center_lat)
    arc_df["source_longitude"] = float(center_lon)
    arc_df["arc_width"] = arc_df["map_weight"].apply(lambda v: 1 + int(float(v) * 4))
    return center_df, arc_df.head(60)


def render_map(records: List[Dict], map_context: Dict | None = None) -> None:
    with st.container(border=True):
        map_context = map_context or {}
        section_heading(
            "Geospatial Map",
            "County Coverage + Radius / Disruption Map",
            "County choropleth fill, weighted company markers, optional hub-to-supplier arcs, and coordinate-source legend.",
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
        df["tooltip_product"] = df.get("product_service", pd.Series(index=df.index)).fillna("Unknown product/service")
        df["tooltip_location"] = (
            df.get("city", pd.Series(index=df.index)).fillna("")
            + ", "
            + df.get("county", pd.Series(index=df.index)).fillna("")
        ).str.strip(", ")
        df["tooltip_coord_source"] = df.get("coordinate_source", pd.Series(index=df.index)).apply(normalize_coordinate_source)
        df["tooltip_weight"] = pd.to_numeric(df["map_weight"], errors="coerce").fillna(0.0).map(lambda v: f"{float(v):.2f}")
        df["tooltip_nearest_peer"] = df.get("nearest_peer_company", pd.Series(index=df.index)).fillna("N/A")
        if "nearest_peer_distance_km" in df.columns:
            df["tooltip_nearest_peer_distance"] = pd.to_numeric(df["nearest_peer_distance_km"], errors="coerce").map(
                lambda v: f"{float(v):.2f} km / {float(v) * 0.621371:.2f} mi" if pd.notna(v) else "N/A"
            )
        else:
            df["tooltip_nearest_peer_distance"] = "N/A"
        if "distance_km" in df.columns:
            df["tooltip_distance"] = pd.to_numeric(df["distance_km"], errors="coerce").map(
                lambda v: f"{float(v):.2f} km / {float(v) * 0.621371:.2f} mi" if pd.notna(v) else "N/A"
            )
        else:
            df["tooltip_distance"] = "N/A"

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
            latitude=float(map_context.get("center_lat") or df["latitude"].mean()),
            longitude=float(map_context.get("center_lon") or df["longitude"].mean()),
            zoom=7.0 if map_context.get("map_mode") == "radius_search" else (6.5 if len(df) > 8 else 7.6),
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

        overlay_geojson = build_county_overlay_geojson(df, map_context)
        county_layer = None
        if overlay_geojson:
            county_layer = pdk.Layer(
                "GeoJsonLayer",
                data=overlay_geojson,
                stroked=True,
                filled=True,
                get_fill_color="properties.fill_color",
                get_line_color="properties.line_color",
                line_width_min_pixels=2,
                pickable=False,
            )

        center_df, arc_df = build_center_and_arc_frames(df, map_context)
        center_layer = None
        arc_layer = None
        if not center_df.empty:
            center_layer = pdk.Layer(
                "ScatterplotLayer",
                data=center_df,
                get_position="[longitude, latitude]",
                get_fill_color=[15, 23, 42, 235],
                get_line_color=[255, 255, 255, 220],
                stroked=True,
                filled=True,
                line_width_min_pixels=2,
                radius_min_pixels=7,
                get_radius="radius",
                pickable=True,
            )
        if not arc_df.empty:
            arc_layer = pdk.Layer(
                "ArcLayer",
                data=arc_df,
                get_source_position="[source_longitude, source_latitude]",
                get_target_position="[longitude, latitude]",
                get_source_color=[18, 137, 127, 155],
                get_target_color="fill_color",
                get_width="arc_width",
                pickable=False,
            )

        radius_layer = None
        if (
            map_context.get("center_lat") is not None
            and map_context.get("center_lon") is not None
            and map_context.get("radius_km") is not None
            and map_context.get("map_mode") == "radius_search"
        ):
            radius_layer = pdk.Layer(
                "GeoJsonLayer",
                data=build_radius_circle_geojson(
                    float(map_context["center_lat"]),
                    float(map_context["center_lon"]),
                    float(map_context["radius_km"]),
                ),
                stroked=True,
                filled=True,
                get_fill_color="properties.fill_color",
                get_line_color=[18, 137, 127, 180],
                line_width_min_pixels=2,
                pickable=False,
            )

        tooltip = {
            "html": (
                "<div style='font-family:Inter, sans-serif; min-width:220px;'>"
                "<div style='font-size:14px; font-weight:800; margin-bottom:8px;'>{tooltip_company}</div>"
                "<div style='font-size:12px; line-height:1.6; color:#e5eef3;'>"
                "<b>Role:</b> {tooltip_role}<br/>"
                "<b>Product:</b> {tooltip_product}<br/>"
                "<b>Location:</b> {tooltip_location}<br/>"
                "<b>Distance:</b> {tooltip_distance}<br/>"
                "<b>Nearest Peer:</b> {tooltip_nearest_peer} ({tooltip_nearest_peer_distance})<br/>"
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
                layers=[
                    layer
                    for layer in [
                        county_layer,
                        radius_layer,
                        arc_layer,
                        heatmap_layer,
                        scatter_layer,
                        center_layer,
                    ]
                    if layer is not None
                ],
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
        render_map_context_summary(result)
        render_map(result.get("retrieved_companies", []) or [], result.get("map_context", {}) or {})
        st.write("")
        render_table(result.get("retrieved_companies", []) or [])
