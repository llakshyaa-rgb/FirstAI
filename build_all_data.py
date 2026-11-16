#!/usr/bin/env python3
"""One-command collection and preparation of FirstAI's sourced data corpus.

Downloads open text/instruction datasets, collects the seven-domain public
data lab, converts records into grounded examples, prepares the base corpus,
and merges balanced instruction data. It does not train a checkpoint.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import threading
import queue
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_step(label, args):
    print("\n" + "=" * 72 + "\n" + label + "\n" + "=" * 72, flush=True)
    print("[START] " + time.strftime("%H:%M:%S") + " | " + " ".join(args), flush=True)
    started = time.monotonic()
    process = subprocess.Popen(
        [sys.executable, "-u", *args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = queue.Queue()

    def read_output():
        try:
            for line in process.stdout:
                lines.put(line.rstrip("\\n"))
        finally:
            lines.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    finished_reading = False
    last_output = time.monotonic()
    while not finished_reading:
        try:
            line = lines.get(timeout=5)
            if line is None:
                finished_reading = True
            else:
                print(line, flush=True)
                last_output = time.monotonic()
        except queue.Empty:
            elapsed = time.monotonic() - started
            quiet = time.monotonic() - last_output
            if process.poll() is not None and not reader.is_alive():
                finished_reading = True
            elif quiet >= 30:
                print(
                    f"[HEARTBEAT] {label} still running | "
                    f"elapsed {elapsed/60:.1f} min | "
                    f"no child output for {quiet:.0f}s",
                    flush=True,
                )
                last_output = time.monotonic()

    return_code = process.wait()
    elapsed = time.monotonic() - started
    if return_code:
        raise SystemExit(
            f"[FAILED] {label} after {elapsed/60:.1f} min "
            f"(exit {return_code}). Earlier outputs are preserved; "
            "inspect the log above, fix the issue, and rerun."
        )
    print(f"[DONE] {label} | elapsed {elapsed/60:.1f} min", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="IND")
    parser.add_argument("--city", default="New Delhi")
    parser.add_argument("--max-per-dataset", type=int, default=10000, help="Maximum training rows per capability dataset.")
    parser.add_argument("--max-base-chars", type=int, default=30_000_000, help="Maximum WikiText characters; bounded for Colab RAM/storage.")
    parser.add_argument("--skip-base-text", action="store_true",
                        help="Skip WikiText/Dolly downloads; reuse existing local files.")
    parser.add_argument("--skip-capability-datasets", action="store_true",
                        help="Skip GSM8K/ARC/SQuAD/MBPP downloads; reuse existing JSONL.")
    parser.add_argument("--per-source-cap", type=int, default=10000)
    parser.add_argument("--skip-conversations", action="store_true",
                        help="Skip multi-turn conversation download; reuse an existing JSONL.")
    parser.add_argument("--conversation-limit", type=int, default=20000)
    parser.add_argument("--skip-live-data", action="store_true",
                        help="Skip live public data collection; reuse an existing data-lab snapshot.")
    args = parser.parse_args()

    if not args.skip_base_text:
        run_step("1/7 — Download base text and instruction data", [
            "download_training_data.py",
            "--wikitext-out", "data/raw/wikitext_train.txt",
            "--max-chars", str(args.max_base_chars),
            "--instructions-out", "data/orbit_instructions_full.jsonl",
        ])
    if not args.skip_capability_datasets:
        run_step("2/7 — Download mathematics, science, reading and coding data", [
            "download_capability_data.py",
            "--max-per-dataset", str(args.max_per_dataset),
        ])
    if not args.skip_conversations:
        run_step("3/7 — Download multi-turn assistant conversations", [
            "download_conversation_data.py", "--limit", str(args.conversation_limit),
        ])
    if not args.skip_live_data:
        run_step("4/7 — Collect live public data across seven domains", [
            "examples/real_data/build_data_lab.py",
            "--country", args.country, "--city", args.city,
        ])
    run_step("5/7 — Convert retrieved real records into sourced examples", [
        "examples/real_data/build_training_examples.py",
    ])
    run_step("6/7 — Prepare deduplicated base-language corpus", [
        "prepare_corpus.py",
    ])
    run_step("7/7 — Merge and balance instruction sources", [
        "merge_instruction_data.py", "--per-source-cap", str(args.per_source_cap),
    ])

    print("\nDATA PREPARATION COMPLETE")
    print("Base corpus: data/input.txt")
    print("Balanced instruction corpus: data/instructions_balanced.jsonl")
    print("Source manifests: data/capabilities/manifest.json and data/instructions_balanced.manifest.json")
    print("Next: review data/SOURCES.md, then train/resume FirstGPT and instruction-tune separately.")
    print("No model checkpoint was changed by this script.")


if __name__ == "__main__":
    main()
