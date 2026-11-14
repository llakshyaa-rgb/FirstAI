#!/usr/bin/env python3
"""Validate the generated multi-domain data pack without network access."""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "generated" / "data_lab"


def validate(root: Path = ROOT) -> dict:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run build_data_lab.py first; manifest.json is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("Manifest has no dataset entries.")
    report = {"valid": True, "status": manifest.get("status"), "datasets": {}, "errors": []}
    for dataset_id, entry in datasets.items():
        if entry.get("status") != "success":
            report["datasets"][dataset_id] = {"status": entry.get("status"), "error": entry.get("error")}
            continue
        json_path = root / (dataset_id + ".json")
        csv_path = root / (dataset_id + ".csv")
        if not json_path.is_file() or not csv_path.is_file():
            report["errors"].append(dataset_id + ": missing JSON or CSV output")
            continue
        bundle = json.loads(json_path.read_text(encoding="utf-8"))
        rows = bundle.get("records")
        if not isinstance(rows, list) or len(rows) != entry.get("row_count"):
            report["errors"].append(dataset_id + ": row count mismatch")
            continue
        if bundle.get("metadata") != entry.get("metadata"):
            report["errors"].append(dataset_id + ": source metadata mismatch")
        if not entry.get("metadata", {}).get("source_url") or not entry.get("retrieved_at_utc"):
            report["errors"].append(dataset_id + ": provenance is incomplete")
        with csv_path.open(newline="", encoding="utf-8") as handle:
            csv_rows = list(csv.DictReader(handle))
        if len(csv_rows) != len(rows):
            report["errors"].append(dataset_id + ": CSV/JSON row count mismatch")
            continue
        # CSV is a portable export; compare decoded cell values, retaining null as blank.
        for index, (json_row, csv_row) in enumerate(zip(rows, csv_rows), 1):
            for key, value in json_row.items():
                actual = csv_row.get(key, "")
                if value is None and actual != "":
                    report["errors"].append(f"{dataset_id}: null changed at row {index}, field {key}")
                    break
                if isinstance(value, (dict, list)):
                    expected = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                    if actual != expected:
                        report["errors"].append(f"{dataset_id}: structured value mismatch at row {index}, field {key}")
                        break
                elif value is not None and str(value) != actual:
                    # CSV serializes numeric JSON values using Python's string representation.
                    if not (isinstance(value, (int, float)) and not isinstance(value, bool) and float(actual) == float(value)):
                        report["errors"].append(f"{dataset_id}: value mismatch at row {index}, field {key}")
                        break
        report["datasets"][dataset_id] = {"status": "valid", "rows": len(rows),
            "source": entry["metadata"].get("source"), "retrieved_at_utc": entry.get("retrieved_at_utc")}
    report["valid"] = not report["errors"]
    report["success_count"] = manifest.get("success_count", 0)
    report["failed_count"] = manifest.get("failed_count", 0)
    if not report["valid"]:
        raise ValueError(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
