from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from backend.ingestion import (
    CITY_COUNTY_FALLBACK,
    DEFAULT_EXCEL_PATH,
    DEFAULT_GEOJSON_PATH,
    extract_city_county,
    extract_city_from_address,
    infer_county_from_point,
    load_county_geometries,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "GNEM_with_updated_location.xlsx"


def _safe_float(value: object) -> Optional[float]:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def build_updated_location(
    address: object,
    latitude: object,
    longitude: object,
    fallback_location: object,
    county_geometries: list[dict],
) -> str:
    city = extract_city_from_address(address)
    fallback_city, fallback_county = extract_city_county(fallback_location)

    if not city:
        city = fallback_city

    county = infer_county_from_point(
        lat=_safe_float(latitude),
        lon=_safe_float(longitude),
        county_geometries=county_geometries,
    )
    if not county and city:
        county = CITY_COUNTY_FALLBACK.get(city.strip().lower())
    if not county:
        county = fallback_county

    if city and county:
        return f"{city}, {county} County"
    if city:
        return str(city)
    if county:
        return f"{county} County"
    return ""


def create_updated_location_workbook(
    excel_path: Path = DEFAULT_EXCEL_PATH,
    geojson_path: Path = DEFAULT_GEOJSON_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    county_geometries = load_county_geometries(geojson_path)

    xls = pd.ExcelFile(excel_path)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)

            if {"Address", "Latitude", "Longitude"}.issubset(df.columns):
                df["Updated Location"] = df.apply(
                    lambda row: build_updated_location(
                        address=row.get("Address"),
                        latitude=row.get("Latitude"),
                        longitude=row.get("Longitude"),
                        fallback_location=row.get("Location"),
                        county_geometries=county_geometries,
                    ),
                    axis=1,
                )

                cols = list(df.columns)
                cols.remove("Updated Location")
                insert_at = cols.index("Location") + 1 if "Location" in cols else len(cols)
                cols.insert(insert_at, "Updated Location")
                df = df[cols]

            df.to_excel(writer, sheet_name=sheet_name, index=False)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an Updated Location column using Address city + county inferred from lat/lon and Georgia county polygons."
    )
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL_PATH, help="Input GNEM workbook path.")
    parser.add_argument("--geojson", type=Path, default=DEFAULT_GEOJSON_PATH, help="Georgia counties GeoJSON path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output workbook path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = create_updated_location_workbook(
        excel_path=args.excel,
        geojson_path=args.geojson,
        output_path=args.output,
    )
    print(f"[updated-location] Workbook written: {out}")
