#!/usr/bin/env python3
"""Merge instruction sources with deduplication and per-source balancing."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def merge(paths, output, per_source_cap=3000):
    groups = defaultdict(list)
    seen = set()
    rejected = 0
    duplicates = 0
    for source_path in paths:
        path = Path(source_path)
        if not path.is_file():
            continue
        fallback_id = path.stem
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            prompt = row.get("prompt")
            response = row.get("response")
            if not prompt and isinstance(row.get("input"), str):
                prompt = str(row.get("instruction") or "Answer using the supplied sourced data.") + "\n\n" + row["input"]
                response = row.get("output")
            if not isinstance(prompt, str) or not prompt.strip() or not isinstance(response, str) or not response.strip():
                rejected += 1
                continue
            digest = (prompt.strip(), response.strip())
            if digest in seen:
                duplicates += 1
                continue
            seen.add(digest)
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            source_id = str(metadata.get("dataset_id") or metadata.get("dataset") or metadata.get("capability") or fallback_id)
            normalized = {"prompt": prompt.strip(), "response": response.strip(),
                          "metadata": {**metadata, "dataset_id": source_id,
                                       "source_file": str(path)}}
            groups[source_id].append(normalized)

    selected = []
    counts = {}
    for source_id in sorted(groups):
        rows = groups[source_id][:per_source_cap]
        selected.extend(rows)
        counts[source_id] = {"available": len(groups[source_id]), "selected": len(rows)}
    if len(selected) < 4:
        raise RuntimeError("Fewer than four valid unique instruction examples were found.")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "output": str(target),
        "examples": len(selected),
        "rejected_invalid_rows": rejected,
        "duplicate_prompt_response_pairs_removed": duplicates,
        "per_source_cap": per_source_cap,
        "sources": counts,
        "note": "Balancing is by dataset ID. Inspect each source license and retain attribution when redistributing.",
    }
    target.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", default=[
        "data/orbit_instructions_full.jsonl",
        "data/orbit_instructions.example.jsonl",
        "data/instructions_capabilities.jsonl",
        "examples/real_data/capability_seed_examples.jsonl",
        "examples/real_data/seed_examples.jsonl",
        "examples/real_data/generated/data_lab_qa_examples.jsonl",
    ])
    parser.add_argument("--output", default="data/instructions_balanced.jsonl")
    parser.add_argument("--per-source-cap", type=int, default=3000)
    args = parser.parse_args()
    if args.per_source_cap < 1:
        parser.error("--per-source-cap must be positive")
    merge(args.inputs, args.output, args.per_source_cap)


if __name__ == "__main__":
    main()
