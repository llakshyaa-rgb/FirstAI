#!/usr/bin/env python3
"""Refresh the FirstAI real-data lab into a timestamped, comparable snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_HISTORY = HERE / "generated" / "data_lab_history"


def digest(path: Path) -> str:
    # Compare actual records, not volatile retrieval timestamps or manifest metadata.
    try:
        bundle = json.loads(path.read_text(encoding="utf-8"))
        payload = bundle.get("records", bundle) if isinstance(bundle, dict) else bundle
        stable = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(stable).hexdigest()
    except (OSError, ValueError, TypeError):
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

def read_manifest(folder: Path):
    path = folder / "manifest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def compare(previous: Path | None, current: Path) -> dict:
    new_manifest = read_manifest(current)
    old_manifest = read_manifest(previous) if previous else None
    if not new_manifest:
        raise RuntimeError("The collector did not produce a readable manifest.")
    changes = {}
    old_datasets = (old_manifest or {}).get("datasets", {})
    for dataset_id, entry in new_manifest.get("datasets", {}).items():
        if entry.get("status") != "success":
            changes[dataset_id] = {
                "status": "unavailable",
                "error": entry.get("error"),
                "previous_snapshot_retained": bool(old_datasets.get(dataset_id, {}).get("status") == "success"),
            }
            continue
        current_file = current / (dataset_id + ".json")
        current_hash = digest(current_file) if current_file.is_file() else None
        old_entry = old_datasets.get(dataset_id, {})
        old_file = (previous / (dataset_id + ".json")) if previous else None
        old_hash = digest(old_file) if old_file and old_file.is_file() else None
        if not old_manifest:
            state = "baseline"
        elif old_entry.get("status") != "success":
            state = "newly_available"
        elif old_hash == current_hash:
            state = "unchanged"
        else:
            state = "changed"
        changes[dataset_id] = {
            "status": state,
            "previous_row_count": old_entry.get("row_count"),
            "current_row_count": entry.get("row_count"),
            "previous_retrieved_at_utc": old_entry.get("retrieved_at_utc"),
            "current_retrieved_at_utc": entry.get("retrieved_at_utc"),
            "previous_sha256": old_hash,
            "current_sha256": current_hash,
        }
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "previous_snapshot": str(previous) if previous else None,
        "current_snapshot": str(current),
        "collector_status": new_manifest.get("status"),
        "success_count": new_manifest.get("success_count", 0),
        "failed_count": new_manifest.get("failed_count", 0),
        "datasets": changes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="IND")
    parser.add_argument("--city", default="New Delhi")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=datetime.now(timezone.utc).year)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    args = parser.parse_args()

    history = args.history.expanduser().resolve()
    history.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    current = history / stamp
    current.mkdir(parents=True, exist_ok=False)
    previous = None
    for candidate in sorted(history.iterdir(), reverse=True):
        if candidate.is_dir() and candidate != current:
            nested = candidate / "data"
            if (nested / "manifest.json").is_file():
                previous = nested
                break
            if (candidate / "manifest.json").is_file():
                previous = candidate
                break

    data_dir = current / "data"
    command = [
        sys.executable, str(HERE / "build_data_lab.py"),
        "--country", args.country, "--city", args.city,
        "--start-year", str(args.start_year), "--end-year", str(args.end_year),
        "--output", str(data_dir),
    ]
    result = subprocess.run(command, cwd=HERE.parent.parent, capture_output=True,
                            text=True, timeout=600, check=False)
    (current / "collector.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (current / "collector.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if not (data_dir / "manifest.json").is_file():
        raise RuntimeError("No manifest was generated. See collector.stderr.txt.")

    sys.path.insert(0, str(HERE))
    from validate_data_lab import validate
    validation = None
    try:
        validation = validate(data_dir)
        validation_status = "valid"
    except Exception as exc:
        validation_status = "invalid_or_partial"
        validation = {"error": type(exc).__name__ + ": " + str(exc)}
    report = compare(previous, data_dir)
    report["collector_exit_code"] = result.returncode
    report["validation_status"] = validation_status
    report["validation"] = validation
    report["stdout_tail"] = result.stdout[-3000:]
    report["stderr_tail"] = result.stderr[-3000:]
    (current / "refresh_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# FirstAI data refresh",
        "",
        "- Retrieved: " + report["generated_at_utc"],
        "- Collector: " + str(report["collector_status"]),
        "- Sources succeeded: " + str(report["success_count"]),
        "- Sources failed: " + str(report["failed_count"]),
        "- Validation: " + validation_status,
        "- Previous snapshot: " + (str(previous) if previous else "none (baseline)"),
        "",
        "| Dataset | Status | Previous rows | Current rows |",
        "|---|---|---:|---:|",
    ]
    for dataset_id, item in report["datasets"].items():
        lines.append("| " + dataset_id + " | " + item["status"] + " | " +
            str(item.get("previous_row_count", "—")) + " | " +
            str(item.get("current_row_count", "—")) + " |")
    lines += ["", "A failed source is marked unavailable, never interpreted as unchanged or zero.",
              "Full provenance and collector output are retained beside this report."]
    (current / "refresh_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "snapshot": str(current), "collector_status": report["collector_status"],
        "success_count": report["success_count"], "failed_count": report["failed_count"],
        "validation_status": validation_status,
        "report": str(current / "refresh_report.md"),
    }, indent=2))
    if report["success_count"] == 0:
        raise SystemExit("All data providers failed; previous snapshots were preserved.")


if __name__ == "__main__":
    main()
