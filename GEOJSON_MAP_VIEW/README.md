# Georgia EV Supply Chain GeoRAG

This app answers EV supply-chain questions with county-aware geospatial retrieval and evidence-grounded RAG over the uploaded Georgia datasets:

- SQL retrieval over GNEM company data (`DuckDB`)
- County and radius search over `Counties_Georgia.geojson`
- Semantic chunk retrieval over facility-aware company chunks (`FAISS` + `sentence-transformers` when available, hashed fallback otherwise)
- Supply-gap and disruption-alternative analytics
- Final answer generation using local Ollama when available, with a retrieval-only fallback if no model is pulled

Architecture:

`User -> Streamlit UI -> FastAPI -> Query Planner -> SQL / Geo / Vector Retrieval -> Ollama LLM`

## Project Structure

```text
project/
|-- backend/
|   |-- ingestion.py
|   |-- main.py
|   |-- query_planner.py
|   |-- rag_pipeline.py
|   |-- spatial_engine.py
|   |-- sql_engine.py
|   `-- vector_engine.py
|-- data/
|   |-- gnem_companies.xlsx
|   |-- GNEM - Auto Landscape Lat Long Updated File (1).xlsx
|   `-- Counties_Georgia.geojson
|-- frontend/
|   `-- app.py
|-- requirements.txt
`-- README.md
```

## Install and Run

From `project/`:

```bash
pip install -r requirements.txt
python backend/ingestion.py
uvicorn backend.main:app --reload
streamlit run frontend/app.py
```

## Ollama Setup (Required)

1. Start Ollama (or ensure it is already running):

```bash
ollama serve
```

2. Pull a model:

```bash
ollama pull gemma3:27b
```

3. (Optional) Set explicit model and endpoint:

```bash
export OLLAMA_MODEL=gemma3:27b
export OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
```

If `OLLAMA_MODEL` is not set, backend auto-selects the best available pulled model from your local Ollama registry, preferring `gemma3:27b`.
If no Ollama model is available, the API still returns retrieval-only answers, map layers, and source citations with `model_used = "retrieval-fallback"`.

### Suggested Models for This RAG Pipeline

- `gemma3:27b` for stronger answer quality if your hardware can run it.
- `qwen3:14b` for a lighter fallback if `gemma3:27b` is too heavy.
- `gpt-oss:20b` for stronger reasoning quality if hardware allows.
- `deepseek-r1:14b` if you want stronger deliberate reasoning style.

For very limited hardware, `qwen3:8b` is the practical minimum. Avoid `tinyllama` for production-quality answers.

If you set a model that does not fit memory, backend will auto-retry smaller locally available models and return `model_used` in the API response.

## Ingestion Pipeline

`backend/ingestion.py` does:

1. Load `GNEM - Auto Landscape Lat Long Updated File (1).xlsx` as the canonical source workbook
2. Normalize company/location columns and preserve duplicate same-site rows when products or roles differ
3. Parse city/county from location/address strings and infer county from point-in-polygon checks against `Counties_Georgia.geojson`
4. Fall back to county centroids when coordinates are missing or land outside the declared Georgia county polygon
5. Write the canonical `companies` table to DuckDB
6. Build facility-aware chunk records:
   - `company_profile`
   - `supply_chain`
   - `products_capabilities`
   - `geo_operations`
   - `resilience_network`
7. Write `company_chunks` table to DuckDB
8. Embed chunk texts and index in FAISS
9. Save chunk metadata JSON for citation tracing

Generated files:

- `data/gnem.duckdb`
- `data/gnem_faiss.index`
- `data/vector_metadata.json`

## API

Endpoint: `POST /chat`

Request:

```json
{
  "question": "Which EV suppliers are near Atlanta?"
}
```

Response includes:

- `answer` (LLM-generated from retrieved chunks)
- `sources` (chunk-level source lines)
- `retrieved_chunks` (full chunk metadata used for context)
- `retrieved_companies` (tabular view)
- `plan` (query planner output)
- `map_context` (query center, county filters, radius, coverage/gap report)
- `model_used` (active Ollama model)

## UI

`frontend/app.py` provides:

- Preset prompt chips for radius search, county filtering, gap analysis, and disruption alternatives
- Citation-highlighted answer panel with `Direct Answer`, `Spatial / Supply-Chain Details`, and `Evidence Gaps`
- County choropleth over Georgia GeoJSON, weighted supplier markers, radius overlays, and hub-to-supplier arcs
- Gap counties highlighted in red when `SUPPLY_CHAIN_GAP_QUERY` is active
- Ranked company table with category, product/service, OEMs, km/mi distance, nearest-peer distance, match reason, map weight, and coordinate provenance
- Retrieved chunk cards for evidence inspection

Map colors indicate coordinate provenance:

- `coordinates_excel:*` = company-level coordinates from the uploaded lat/long workbook
- `source_excel` = coordinates found directly in the main GNEM sheet
- `county_centroid` = estimated county centroid fallback

## Notes

- For best natural-language synthesis, use a stronger local model than `tinyllama` (for example `qwen3:14b`).
- One row in the uploaded workbook (`Valeo`) does not contain location or coordinates, so it is retained in retrieval but cannot be plotted until those fields are supplied.
