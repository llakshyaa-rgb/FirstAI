#!/usr/bin/env python3
"""Download multi-turn assistant conversations and convert them to FirstGPT SFT rows."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

DATASET = "HuggingFaceH4/ultrachat_200k"
SPLIT = "train_sft"
DATASET_URL = "https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k"
LICENSE = "MIT"

def _messages(row):
    value = row.get("messages") or row.get("conversation") or []
    cleaned = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or item.get("from") or "").strip().lower()
        text = item.get("content")
        if text is None:
            text = item.get("value")
        if role in {"human", "user"}:
            role = "user"
        elif role in {"assistant", "gpt"}:
            role = "assistant"
        elif role != "system":
            continue
        if isinstance(text, str) and text.strip():
            cleaned.append((role, text.strip()))
    return cleaned

def convert_conversation(messages, source_index, max_context_chars=6000):
    prior = []
    for role, text in messages:
        if role == "assistant" and any(r == "user" for r, _ in prior):
            lines = []
            for previous_role, previous_text in prior[-12:]:
                label = {"user": "Owner", "assistant": "Orbit", "system": "System"}[previous_role]
                lines.append(f"{label}: {previous_text}")
            prompt = "\n".join(lines)
            if len(prompt) > max_context_chars:
                prompt = prompt[-max_context_chars:]
            prompt += "\nOrbit:"
            if len(text) > 1800:
                prior.append((role, text))
                continue
            response = "Orbit: " + text
            digest = hashlib.sha256((prompt + "\0" + response).encode("utf-8")).hexdigest()
            yield {"prompt": prompt, "response": response, "metadata": {
                "dataset_id": "ultrachat_200k", "dataset": DATASET, "split": SPLIT,
                "license_declared_by_dataset_card": LICENSE, "source_url": DATASET_URL,
                "source_record_index": source_index, "turn_hash": digest,
                "purpose": "multi-turn conversational instruction tuning",
            }}
        prior.append((role, text))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=30000)
    parser.add_argument("--output", type=Path, default=Path("data/instructions_conversations.jsonl"))
    parser.add_argument("--max-context-chars", type=int, default=2500)
    args = parser.parse_args()
    if args.limit < 100 or args.max_context_chars < 256:
        parser.error("--limit must be >=100 and --max-context-chars >=256")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install optional dependency: python -m pip install 'datasets>=3.0'") from exc
    print(f"Loading {DATASET} [{SPLIT}] — training split only", flush=True)
    dataset = load_dataset(DATASET, split=SPLIT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written, seen = 0, set()
    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(dataset):
            messages = _messages(row)
            for example in convert_conversation(messages, index, args.max_context_chars):
                key = (example["prompt"], example["response"])
                if key in seen:
                    continue
                seen.add(key)
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
                written += 1
                if written % 1000 == 0:
                    print(f"Prepared {written:,} assistant turns", flush=True)
                if written >= args.limit:
                    break
            if written >= args.limit:
                break
    if written < 100:
        args.output.unlink(missing_ok=True)
        raise RuntimeError(f"Only {written} usable assistant turns; refusing a weak dataset.")
    manifest = {"dataset": DATASET, "split": SPLIT, "license_declared_by_dataset_card": LICENSE,
        "source_url": DATASET_URL, "examples": written, "output": str(args.output),
        "deduplication": "exact prompt/response pairs",
        "note": "Review current dataset card and terms before redistribution; data is not verified factual ground truth."}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)

if __name__ == "__main__":
    main()
