# Hybrid Geospatial Retrieval and GeoJSON-Based Supplier Mapping

## Poster Visual Layout

| **1. Study Context + Data** | **2. Methods + System Design** | **3. Outputs + Interpretation** |
|---|---|---|
| **Research Goal**<br><br>Develop a GeoJSON-supported geospatial retrieval system that answers questions about Georgia EV supply-chain companies and maps relevant suppliers interactively.<br><br>**Data Sources**<br><br>`gnem_companies.xlsx`<br>GNEM company records, OEM relationships, product/service data, employment, and location fields.<br><br>`GNEM - Auto Landscape Lat Long Updated File (1).xlsx`<br>Latitude/longitude enrichment for company locations.<br><br>`Counties_Georgia.geojson`<br>Georgia county boundary polygons and county fallback geometry.<br><br>**Study Region**<br><br>Georgia, USA county-level spatial framework.<br><br>**Dataset Summary**<br><br>\| Item \| Value \|<br>\|---\|---\|<br>\| Companies \| 207 \|<br>\| Semantic chunks \| 828 \|<br>\| GeoJSON counties \| 159 \|<br>\| Exact/enriched coords \| 202 \|<br>\| County-centroid fallback \| 4 \|<br>\| Missing coords \| 1 \| | **Method Pipeline**<br><br>1. Clean company Excel data<br>2. Extract city/county from location text<br>3. Attach coordinates from enrichment workbook<br>4. Use GeoJSON county centroids if exact coordinates are missing<br>5. Store company rows in DuckDB<br>6. Convert each company into 4 semantic chunks<br>7. Build FAISS vector index<br>8. Route user questions through SQL + Geo + Vector retrieval<br>9. Validate evidence support<br>10. Generate answer using local Ollama LLM<br>11. Render GeoJSON county map + company heatmap + point markers<br><br>**GeoJSON Role**<br><br>\| Function \| Implementation \|<br>\|---\|---\|<br>\| Boundary display \| Pydeck `GeoJsonLayer` \|<br>\| Fallback geocoding \| County centroid from polygons \|<br>\| City/county resolution \| County centroid + city aliases \|<br><br>**Retrieval Logic**<br><br>\| Query Type \| Engine \|<br>\|---\|---\|<br>\| OEM / industry / top employment \| DuckDB SQL \|<br>\| Near / within radius / coordinates \| geopy spatial search \|<br>\| Supplier / battery / product semantics \| FAISS vector search \|<br>\| Mixed spatial + business query \| Hybrid combination \| | **Frontend Output**<br><br>**A. LLM Answer Panel**<br>Natural-language answer generated only from retrieved chunks, with chunk citations such as `[C1]`.<br><br>**B. Evidence Tables**<br>Retrieved chunk table + retrieved company table for transparency.<br><br>**C. Supplier Map**<br>Georgia county boundaries + heatmap + point markers.<br><br>**Map Legend**<br><br>\| Color \| Coordinate Meaning \|<br>\|---\|---\|<br>\| Teal \| Coordinate workbook match \|<br>\| Blue \| Source Excel coordinate \|<br>\| Orange \| GeoJSON county centroid fallback \|<br>\| Red \| Missing coordinate \|<br><br>**Map Weighting**<br><br>`map_weight = 0.30 relevance + 0.25 query_match + 0.20 proximity + 0.15 business_priority + 0.10 metric_score`<br><br>**Interpretation**<br><br>Larger and hotter markers indicate companies that are more relevant to the query, closer to the target location, and more important from a supply-chain role perspective. |

## Architecture Diagram

```mermaid
flowchart LR
    A[GNEM Excel<br/>+ Coordinate Excel<br/>+ Georgia County GeoJSON]
    B[Data Cleaning<br/>+ City/County Parsing<br/>+ Coordinate Enrichment]
    C[DuckDB<br/>Companies Table]
    D[Chunk Builder<br/>4 Text Chunks per Company]
    E[FAISS Vector Index<br/>+ Chunk Metadata]
    F[Streamlit Chat UI]
    G[FastAPI Backend<br/>/chat Endpoint]
    H[Rule-Based Query Planner]
    I[SQL Retrieval<br/>DuckDB]
    J[Geo Retrieval<br/>geopy Distance Search]
    K[Vector Retrieval<br/>FAISS Hybrid Score]
    L[Evidence Merge<br/>+ Support Guard]
    M[Ollama LLM<br/>Citation-Based Answer]
    N[Pydeck Map<br/>GeoJsonLayer + Heatmap + Scatterplot]

    A --> B
    B --> C
    B --> D
    D --> E
    F --> G
    G --> H
    H --> I
    H --> J
    H --> K
    I --> L
    J --> L
    K --> L
    L --> M
    L --> N
    M --> F
    N --> F
```

## Poster-Ready Methods Text

> We developed a hybrid geospatial retrieval and mapping pipeline for Georgia EV supply-chain company analysis. Company data were cleaned from Excel records, enriched with latitude/longitude from an external coordinate workbook, and completed using county-centroid fallback locations derived from Georgia county GeoJSON polygons. Each company was converted into four semantic text chunks describing profile, supply-chain links, product capabilities, and geographic operations. DuckDB was used for structured SQL retrieval, FAISS for semantic chunk retrieval, and geopy for radius-based distance filtering. A rule-based planner selected SQL, vector, geospatial, or hybrid retrieval depending on the user question. Retrieved evidence was checked before answer generation to reduce unsupported responses. A local Ollama LLM then generated citation-based answers from retrieved chunks only, while Streamlit and Pydeck rendered county GeoJSON boundaries, a supplier heatmap, and point markers weighted by retrieval relevance, query match, proximity, business priority, and metric score.

## Suggested Poster Figure Captions

**Figure 1. Study region and supplier map.** Georgia county boundaries from GeoJSON overlaid with retrieved supplier locations. Marker size and heat intensity represent query-aware relevance and proximity.

**Figure 2. System architecture.** Hybrid retrieval workflow integrating DuckDB SQL search, FAISS semantic retrieval, geopy spatial filtering, and Ollama-based answer generation.

**Figure 3. Evidence-grounded QA interface.** The UI displays the generated answer, supporting retrieved chunks, structured company records, and a map visualization for spatial interpretation.

## Strengths, Limitations, and Future Work

| **Strengths** | **Limitations** | **Future Work** |
|---|---|---|
| Integrates structured, semantic, and spatial retrieval in one pipeline<br><br>GeoJSON supports both map rendering and fallback geolocation<br><br>Chunk citations and evidence tables improve transparency | County-centroid fallback is approximate<br><br>Current map does not yet include highways, rail, ports, or logistics corridors<br><br>Query planner is rule-based and may miss ambiguous phrasing | Add road, rail, and port infrastructure layers<br><br>Add radius-circle overlays and logistics corridors<br><br>Improve geocoding accuracy and evaluate retrieval performance quantitatively |

## One-Box Summary for Poster

> **Core contribution:** A GeoJSON-enabled hybrid geospatial RAG system that links Georgia supply-chain company records to county geometries, retrieves companies using SQL + vector + spatial search, and presents results through both evidence-grounded text answers and an interpretable supplier map.
