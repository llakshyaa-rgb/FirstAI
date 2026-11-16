"""Stream and incrementally checkpoint public text/instruction datasets.

Writes directly to the workspace (no Drive mount). Partial outputs are retained
and resumed on rerun. Progress is printed periodically so notebook cells are
observable instead of appearing frozen during a large dataset download.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def _count_jsonl(path):
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def download_wikitext(output, config="wikitext-103-raw-v1", max_chars=5_000_000,
                      progress_every=10_000, resume=True):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the optional datasets package: pip install datasets") from exc

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    if target.exists() and not resume:
        target.unlink()
    if target.exists() and target.stat().st_size:
        existing = target.stat().st_size
        print(f"WikiText already present: {existing:,} bytes; keeping it.", flush=True)
        return target

    written = partial.stat().st_size if resume and partial.exists() else 0
    if not resume:
        partial.unlink(missing_ok=True)
    print(f"Streaming Salesforce/wikitext [{config}] — target up to {max_chars:,} chars; resuming at {written:,} bytes.", flush=True)
    dataset = load_dataset("Salesforce/wikitext", config, split="train", streaming=True)
    started = time.monotonic()
    last_report = written
    mode = "a" if written else "w"
    with partial.open(mode, encoding="utf-8") as handle:
        for index, row in enumerate(dataset):
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            remaining = max_chars - written
            if remaining <= 0:
                break
            text = text[:remaining]
            chunk = text + "\\n\\n"
            handle.write(chunk)
            written += len(chunk)
            if written - last_report >= progress_every:
                handle.flush()
                elapsed = max(time.monotonic() - started, 0.001)
                print(f"WikiText: {written:,}/{max_chars:,} chars | row {index:,} | {written / elapsed:,.0f} chars/sec", flush=True)
                last_report = written
            if written >= max_chars:
                break
        handle.flush()

    if written < 20_000:
        raise RuntimeError(f"Downloaded corpus is unexpectedly small: {written} characters. Partial data retained at {partial}.")
    partial.replace(target)
    manifest = {
        "dataset": "Salesforce/wikitext", "config": config, "split": "train",
        "license": "CC-BY-SA-3.0 and GFDL (verify current dataset card)",
        "source_url": "https://huggingface.co/datasets/Salesforce/wikitext",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "characters_written": written, "max_chars": max_chars,
    }
    target.with_suffix(target.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\\n", encoding="utf-8")
    print(f"WikiText COMPLETE: {target} | {written:,} characters", flush=True)
    return target


def download_dolly(output, max_examples=15_000, extra_examples=None,
                   progress_every=1000, resume=True):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the optional datasets package: pip install datasets") from exc
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    if target.exists() and target.stat().st_size:
        print(f"Dolly output already present: {_count_jsonl(target):,} examples; keeping it.", flush=True)
        return target
    if not resume:
        partial.unlink(missing_ok=True)
    count = _count_jsonl(partial) if partial.exists() else 0
    print(f"Streaming databricks-dolly-15k; resuming at {count:,} examples.", flush=True)
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
    started = time.monotonic()
    mode = "a" if count else "w"
    with partial.open(mode, encoding="utf-8") as handle:
        for index, row in enumerate(dataset):
            if count >= max_examples:
                break
            instruction = str(row.get("instruction", "")).strip()
            response = str(row.get("response", "")).strip()
            context = str(row.get("context", "")).strip()
            if not instruction or not response:
                continue
            prompt = instruction + ("\\n\\nReference context:\\n" + context if context else "")
            handle.write(json.dumps({
                "prompt": prompt, "response": response,
                "metadata": {
                    "dataset_id": "dolly15k", "dataset": "databricks/databricks-dolly-15k",
                    "split": "train", "license": "CC-BY-SA-3.0",
                    "source_url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
                },
            }, ensure_ascii=False) + "\\n")
            count += 1
            if count % progress_every == 0:
                handle.flush()
                elapsed = max(time.monotonic() - started, 0.001)
                print(f"Dolly: {count:,}/{max_examples:,} examples | row {index:,} | {count / elapsed:.1f} examples/sec", flush=True)
        handle.flush()
    if extra_examples:
        extra_path = Path(extra_examples)
        if extra_path.is_file():
            with partial.open("a", encoding="utf-8") as dest, extra_path.open(encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        dest.write(line)
                        count += 1
    if count < 4:
        raise RuntimeError(f"Instruction dataset download produced only {count} examples; partial retained at {partial}.")
    partial.replace(target)
    manifest = {
        "dataset": "databricks/databricks-dolly-15k", "split": "train",
        "license": "CC-BY-SA-3.0",
        "source_url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_examples_including_extra": count,
        "note": "Orbit extra examples are appended after Dolly; inspect their individual metadata/source file.",
    }
    target.with_suffix(target.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\\n", encoding="utf-8")
    print(f"Instruction data COMPLETE: {target} | {count:,} examples", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description="Stream open text/instruction data with visible progress and resumable partial files.")
    parser.add_argument("--wikitext-out", default="data/wikitext_train.txt")
    parser.add_argument("--wikitext-config", choices=("wikitext-2-raw-v1", "wikitext-103-raw-v1"), default="wikitext-103-raw-v1")
    parser.add_argument("--max-chars", type=int, default=5_000_000)
    parser.add_argument("--instructions-out", default="data/orbit_instructions_full.jsonl")
    parser.add_argument("--max-instructions", type=int, default=15_000)
    parser.add_argument("--extra-instructions", default="data/orbit_instructions.example.jsonl")
    parser.add_argument("--skip-wikitext", action="store_true")
    parser.add_argument("--skip-instructions", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="Discard partial files and start downloads from the beginning.")
    args = parser.parse_args()
    if args.max_chars < 20_000 or args.max_instructions < 4:
        parser.error("max-chars must be >= 20000 and max-instructions >= 4")
    if not args.skip_wikitext:
        download_wikitext(args.wikitext_out, args.wikitext_config, args.max_chars, resume=not args.no_resume)
    if not args.skip_instructions:
        download_dolly(args.instructions_out, args.max_instructions, args.extra_instructions, resume=not args.no_resume)


if __name__ == "__main__":
    main()
