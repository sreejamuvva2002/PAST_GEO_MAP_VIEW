from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "gnem.duckdb"


class SQLEngine:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"DuckDB file not found: {self.db_path}")

    def _query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        with duckdb.connect(str(self.db_path), read_only=True) as con:
            return con.execute(sql, params or []).fetchdf()

    @staticmethod
    def _select_columns(extra_columns: str = "") -> str:
        base = """
                company, category, industry_group, location, address, city, county,
                ev_supply_chain_role, primary_oems, supplier_or_affiliation_type,
                employment, product_service, ev_battery_relevant, primary_facility_type,
                latitude, longitude, coordinate_source
        """
        return f"{base}, {extra_columns}" if extra_columns else base

    def list_company_names(self) -> List[str]:
        df = self._query(
            """
            SELECT DISTINCT company
            FROM companies
            WHERE company IS NOT NULL AND TRIM(company) != ''
            ORDER BY company
            """
        )
        return df["company"].dropna().astype(str).tolist() if not df.empty else []

    def get_companies_by_oem(self, oem_name: str) -> pd.DataFrame:
        pattern = f"%{oem_name.lower()}%"
        sql = """
            SELECT
                company, category, industry_group, location, address, city, county,
                ev_supply_chain_role, primary_oems, supplier_or_affiliation_type,
                employment, product_service, ev_battery_relevant, primary_facility_type,
                latitude, longitude, coordinate_source
            FROM companies
            WHERE LOWER(COALESCE(primary_oems, '')) LIKE ?
            ORDER BY employment DESC NULLS LAST, company
        """
        return self._query(sql, [pattern])

    def get_top_companies_by_metric(self, metric: str, limit: int = 10) -> pd.DataFrame:
        metric_map: Dict[str, str] = {
            "employment": "employment",
            "employees": "employment",
        }
        metric_key = metric.strip().lower()
        if metric_key not in metric_map:
            valid = ", ".join(sorted(metric_map.keys()))
            raise ValueError(f"Unsupported metric '{metric}'. Valid metrics: {valid}")

        column = metric_map[metric_key]
        sql = f"""
            SELECT
                company, category, industry_group, location, address, city, county,
                ev_supply_chain_role, primary_oems, supplier_or_affiliation_type,
                employment, product_service, ev_battery_relevant, primary_facility_type,
                latitude, longitude, coordinate_source,
                {column} AS metric_value
            FROM companies
            WHERE {column} IS NOT NULL
            ORDER BY {column} DESC NULLS LAST, company
            LIMIT ?
        """
        return self._query(sql, [int(limit)])

    def get_companies_by_industry(self, industry_group: str) -> pd.DataFrame:
        pattern = f"%{industry_group.lower()}%"
        sql = """
            SELECT
                company, category, industry_group, location, address, city, county,
                ev_supply_chain_role, primary_oems, supplier_or_affiliation_type,
                employment, product_service, ev_battery_relevant, primary_facility_type,
                latitude, longitude, coordinate_source
            FROM companies
            WHERE LOWER(COALESCE(industry_group, '')) LIKE ?
            ORDER BY employment DESC NULLS LAST, company
        """
        return self._query(sql, [pattern])

    def get_companies_by_name(self, company_name: str) -> pd.DataFrame:
        pattern = f"%{company_name.lower()}%"
        sql = """
            SELECT
                company, category, industry_group, location, address, city, county,
                ev_supply_chain_role, primary_oems, supplier_or_affiliation_type,
                employment, product_service, ev_battery_relevant, primary_facility_type,
                latitude, longitude, coordinate_source
            FROM companies
            WHERE LOWER(COALESCE(company, '')) LIKE ?
            ORDER BY employment DESC NULLS LAST, company, location
        """
        return self._query(sql, [pattern])

    def search_companies(
        self,
        oem_name: Optional[str] = None,
        category_term: Optional[str] = None,
        capability_term: Optional[str] = None,
        city_term: Optional[str] = None,
        county_names: Optional[List[str]] = None,
        company_term: Optional[str] = None,
        limit: int = 50,
    ) -> pd.DataFrame:
        clauses = []
        params: list = []

        if oem_name:
            clauses.append("LOWER(COALESCE(primary_oems, '')) LIKE ?")
            params.append(f"%{oem_name.lower()}%")
        if category_term:
            clauses.append("LOWER(COALESCE(category, '')) LIKE ?")
            params.append(f"%{category_term.lower()}%")
        if capability_term:
            clauses.append(
                "("
                "LOWER(COALESCE(industry_group, '')) LIKE ? OR "
                "LOWER(COALESCE(product_service, '')) LIKE ? OR "
                "LOWER(COALESCE(ev_supply_chain_role, '')) LIKE ?"
                ")"
            )
            params.extend([f"%{capability_term.lower()}%"] * 3)
        if city_term:
            clauses.append(
                "("
                "LOWER(COALESCE(city, '')) LIKE ? OR "
                "LOWER(COALESCE(location, '')) LIKE ? OR "
                "LOWER(COALESCE(county, '')) LIKE ?"
                ")"
            )
            params.extend([f"%{city_term.lower()}%"] * 3)
        if county_names:
            placeholders = ", ".join(["?"] * len(county_names))
            clauses.append(f"LOWER(COALESCE(county, '')) IN ({placeholders})")
            params.extend([county.lower() for county in county_names])
        if company_term:
            clauses.append("LOWER(COALESCE(company, '')) LIKE ?")
            params.append(f"%{company_term.lower()}%")

        where_sql = " AND ".join(clauses) if clauses else "1=1"
        sql = f"""
            SELECT
                company, category, industry_group, location, address, city, county,
                ev_supply_chain_role, primary_oems, supplier_or_affiliation_type,
                employment, product_service, ev_battery_relevant, primary_facility_type,
                latitude, longitude, coordinate_source
            FROM companies
            WHERE {where_sql}
            ORDER BY employment DESC NULLS LAST, company
            LIMIT ?
        """
        params.append(int(limit))
        return self._query(sql, params)
