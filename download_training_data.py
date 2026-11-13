"""Download public text datasets directly into the Colab/local workspace.

No Drive mount or manual upload is used. Dataset sources and licenses are
recorded in data/SOURCES.md. Review each dataset's card and terms before use.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def download_wikitext(output, config="wikitext-2-raw-v1", max_chars=1_000_000):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the optional datasets package: pip install datasets") from exc
    dataset = load_dataset("Salesforce/wikitext", config, split="train")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in dataset:
            text = row.get("text", "").strip()
            if not text:
                continue
            remaining = max_chars - written
            if remaining <= 0:
                break
            text = text[:remaining]
            handle.write(text + "\n\n")
            written += len(text) + 2
    if written < 20_000:
        raise RuntimeError(f"Downloaded corpus is unexpectedly small: {written} characters")
    print(f"WikiText saved: {target} | {written:,} characters | config={config}")
    return target


def download_dolly(output, max_examples=15_000, extra_examples=None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the optional datasets package: pip install datasets") from exc
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in dataset:
            if count >= max_examples:
                break
            instruction = str(row.get("instruction", "")).strip()
            response = str(row.get("response", "")).strip()
            context = str(row.get("context", "")).strip()
            if not instruction or not response:
                continue
            prompt = instruction
            if context:
                prompt += "\n\nReference context:\n" + context
            handle.write(json.dumps({"prompt": prompt, "response": response},
                                     ensure_ascii=False) + "\n")
            count += 1
    if extra_examples:
        extra_path = Path(extra_examples)
        if extra_path.is_file():
            with extra_path.open(encoding="utf-8") as source, target.open("a", encoding="utf-8") as dest:
                for line in source:
                    if line.strip():
                        dest.write(line)
                        count += 1
    if count < 4:
        raise RuntimeError("Instruction dataset download produced too few examples")
    print(f"Instruction data saved: {target} | {count:,} examples")
    return target


def main():
    parser = argparse.ArgumentParser(description="Download open text and instruction datasets without manual uploads.")
    parser.add_argument("--wikitext-out", default="data/wikitext_train.txt")
    parser.add_argument("--wikitext-config", choices=("wikitext-2-raw-v1", "wikitext-103-raw-v1"),
                        default="wikitext-2-raw-v1")
    parser.add_argument("--max-chars", type=int, default=1_000_000)
    parser.add_argument("--instructions-out", default="data/orbit_instructions_full.jsonl")
    parser.add_argument("--max-instructions", type=int, default=15_000)
    parser.add_argument("--extra-instructions", default="data/orbit_instructions.example.jsonl")
    parser.add_argument("--skip-wikitext", action="store_true")
    parser.add_argument("--skip-instructions", action="store_true")
    args = parser.parse_args()
    if args.max_chars < 20_000 or args.max_instructions < 4:
        parser.error("max-chars must be >= 20000 and max-instructions >= 4")
    if not args.skip_wikitext:
        download_wikitext(args.wikitext_out, args.wikitext_config, args.max_chars)
    if not args.skip_instructions:
        download_dolly(args.instructions_out, args.max_instructions, args.extra_instructions)


if __name__ == "__main__":
    main()
