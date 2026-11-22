#!/usr/bin/env python3
"""Build a shuffled, source-balanced instruction dataset without overwriting the old one."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# These are maximums, not targets. All available examples are retained below
# each cap. Conversation and general instruction data remain the largest share;
# narrow passage QA and school-science QA are deliberately reduced.
DEFAULT_SOURCE_CAPS = {
    "ultrachat_200k": 20000,
    "dolly15k": 14985,
    "squad": 3000,
    "gsm8k": 5000,
    "arc": 1000,
    "mbpp": 500,
    "orbit_instructions_full": 500,
    "seed_examples": 500,
}
DEFAULT_OTHER_CAP = 500


def normalize_pair(prompt, response):
    return (
        " ".join(prompt.casefold().split()),
        " ".join(response.casefold().split()),
    )


def merge(paths, output, source_caps=None, other_cap=DEFAULT_OTHER_CAP, seed=20261122):
    rng = random.Random(seed)
    groups = defaultdict(list)
    seen_pairs = set()
    rejected = 0
    duplicates = 0
    source_files = Counter()

    for source_path in paths:
        path = Path(source_path)
        if not path.is_file():
            print(f"[SKIP] Missing source: {path}")
            continue

        fallback_id = path.stem
        with path.open(encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc

                prompt = row.get("prompt")
                response = row.get("response")
                if not prompt and isinstance(row.get("input"), str):
                    prompt = (
                        str(row.get("instruction") or "Answer using the supplied sourced data.")
                        + "\n\n" + row["input"]
                    )
                    response = row.get("output")

                if (
                    not isinstance(prompt, str) or not prompt.strip()
                    or not isinstance(response, str) or not response.strip()
                ):
                    rejected += 1
                    continue

                prompt = prompt.strip()
                response = response.strip()
                if len(prompt) < 3 or len(response) < 2:
                    rejected += 1
                    continue

                digest = normalize_pair(prompt, response)
                if digest in seen_pairs:
                    duplicates += 1
                    continue
                seen_pairs.add(digest)

                metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                source_id = str(
                    metadata.get("dataset_id")
                    or metadata.get("dataset")
                    or metadata.get("capability")
                    or fallback_id
                )
                normalized = {
                    "prompt": prompt,
                    "response": response,
                    "metadata": {
                        **metadata,
                        "dataset_id": source_id,
                        "source_file": str(path),
                    },
                }
                groups[source_id].append(normalized)
                source_files[str(path)] += 1

    if not groups:
        raise RuntimeError("No valid instruction examples were found.")

    caps = dict(DEFAULT_SOURCE_CAPS)
    if source_caps:
        caps.update(source_caps)

    selected = []
    counts = {}
    for source_id in sorted(groups):
        available = groups[source_id]
        cap = caps.get(source_id, other_cap)
        if len(available) > cap:
            chosen = rng.sample(available, cap)
        else:
            chosen = list(available)
        rng.shuffle(chosen)
        selected.extend(chosen)
        counts[source_id] = {
            "available": len(available),
            "cap": cap,
            "selected": len(chosen),
        }

    rng.shuffle(selected)
    if len(selected) < 1000:
        raise RuntimeError(
            f"Refusing to replace the dataset with only {len(selected)} examples."
        )

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as destination:
        for row in selected:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(target)

    manifest = {
        "schema_version": 2,
        "output": str(target),
        "examples": len(selected),
        "seed": seed,
        "rejected_invalid_rows": rejected,
        "normalized_duplicate_pairs_removed": duplicates,
        "source_caps": caps,
        "other_source_cap": other_cap,
        "sources": counts,
        "source_files": dict(source_files),
        "shuffle": "deterministic global shuffle after per-source random sampling",
        "note": (
            "The existing instructions_balanced.jsonl is not overwritten by default. "
            "Review source dataset cards and license obligations before redistribution."
        ),
    }
    manifest_path = target.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def parse_caps(values):
    caps = {}
    for value in values or []:
        if "=" not in value:
            raise argparse.ArgumentTypeError("Source caps must use SOURCE=COUNT")
        name, raw_count = value.split("=", 1)
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid cap: {value}") from exc
        if not name or count < 1:
            raise argparse.ArgumentTypeError(f"Invalid cap: {value}")
        caps[name] = count
    return caps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", default=[
        "data/orbit_instructions_full.jsonl",
        "data/instructions_conversations.jsonl",
        "data/orbit_instructions.example.jsonl",
        "data/instructions_capabilities.jsonl",
        "examples/real_data/capability_seed_examples.jsonl",
        "examples/real_data/seed_examples.jsonl",
        "examples/real_data/generated/data_lab_qa_examples.jsonl",
    ])
    parser.add_argument("--output", default="data/instructions_rebuilt.jsonl")
    parser.add_argument("--other-source-cap", type=int, default=DEFAULT_OTHER_CAP)
    parser.add_argument("--source-cap", action="append", default=[], metavar="SOURCE=COUNT")
    parser.add_argument("--seed", type=int, default=20261122)
    args = parser.parse_args()

    if args.other_source_cap < 1:
        parser.error("--other-source-cap must be positive")
    try:
        caps = parse_caps(args.source_cap)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    merge(args.inputs, args.output, caps, args.other_source_cap, args.seed)


if __name__ == "__main__":
    main()
