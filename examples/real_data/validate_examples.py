#!/usr/bin/env python3
"""Validate consistency and provenance of generated World Bank demo artifacts.

This checks local artifact integrity only; it does not independently certify the
World Bank source data or replace a review of the source metadata.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "generated"
INDICATORS = {"population", "gdp_current_usd", "life_expectancy", "internet_users_pct"}
EXPECTED_CODES = {"SP.POP.TOTL", "NY.GDP.MKTP.CD", "SP.DYN.LE00.IN", "IT.NET.USER.ZS"}


def validate(root: Path = ROOT) -> dict:
    required = ["manifest.json", "world_bank_data.json", "world_bank_data.csv",
                "dashboard.html", "research_report.md"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing generated artifacts: " + ", ".join(missing))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    bundle = json.loads((root / "world_bank_data.json").read_text(encoding="utf-8"))
    if bundle.get("manifest") != manifest:
        raise ValueError("JSON bundle manifest does not match manifest.json")
    records = bundle.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("The JSON dataset must contain at least one observation row.")
    if manifest.get("source") != "World Bank Open Data API" or not manifest.get("retrieved_at_utc"):
        raise ValueError("Source provenance or retrieval timestamp is missing.")
    codes = {item.get("code") for item in manifest.get("indicators", [])}
    if codes != EXPECTED_CODES:
        raise ValueError("Indicator codes do not match the documented data contract.")
    if manifest.get("record_count") != len(records):
        raise ValueError("Manifest row count does not match JSON records.")

    fields = ["country_code", "year", *sorted(INDICATORS)]
    with (root / "world_bank_data.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        csv_rows = list(reader)
        if set(reader.fieldnames or []) != set(fields):
            raise ValueError("CSV columns do not match the expected schema.")
    if len(csv_rows) != len(records):
        raise ValueError("CSV and JSON row counts differ.")

    seen_years = set()
    year_order = []
    for index, (json_row, csv_row) in enumerate(zip(records, csv_rows), 1):
        year = json_row.get("year")
        if type(year) is not int or year in seen_years:
            raise ValueError(f"Invalid or duplicate year in row {index}.")
        seen_years.add(year)
        year_order.append(year)
        if json_row.get("country_code") != manifest.get("country_code"):
            raise ValueError(f"Country code mismatch in row {index}.")
        for key in INDICATORS:
            value = json_row.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise ValueError(f"Non-numeric observation for {key} in row {index}.")
            csv_value = csv_row.get(key)
            if value is None and csv_value not in ("", None):
                raise ValueError(f"CSV changed a missing value for {key} in row {index}.")
            if value is not None and float(csv_value) != float(value):
                raise ValueError(f"CSV/JSON value mismatch for {key} in row {index}.")
    if year_order != sorted(year_order):
        raise ValueError("Year ordering is inconsistent.")
    dashboard = (root / "dashboard.html").read_text(encoding="utf-8")
    report = (root / "research_report.md").read_text(encoding="utf-8")
    if "World Bank Open Data" not in dashboard or "data.worldbank.org" not in report:
        raise ValueError("Source attribution is missing from the dashboard or report.")
    return {"valid": True, "rows": len(records), "country": manifest["country_code"],
            "retrieved_at_utc": manifest["retrieved_at_utc"],
            "missing_observations": sum(1 for row in records for key in INDICATORS if row.get(key) is None),
            "optional_xlsx_present": (root / "world_bank_data.xlsx").is_file()}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
