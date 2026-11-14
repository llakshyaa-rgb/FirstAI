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
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_step(label, args):
    print("\n" + "=" * 72 + "\n" + label + "\n" + "=" * 72, flush=True)
    result = subprocess.run([sys.executable, *args], cwd=ROOT)
    if result.returncode:
        raise SystemExit(f"Stopped at '{label}' (exit {result.returncode}). Earlier outputs are preserved; fix the issue and rerun.")
    

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="IND")
    parser.add_argument("--city", default="New Delhi")
    parser.add_argument("--max-per-dataset", type=int, default=3000)
    parser.add_argument("--max-base-chars", type=int, default=5_000_000)
    parser.add_argument("--skip-base-text", action="store_true",
                        help="Skip WikiText/Dolly downloads; reuse existing local files.")
    parser.add_argument("--skip-capability-datasets", action="store_true",
                        help="Skip GSM8K/ARC/SQuAD/MBPP downloads; reuse existing JSONL.")
    parser.add_argument("--skip-live-data", action="store_true",
                        help="Skip live public data collection; reuse an existing data-lab snapshot.")
    args = parser.parse_args()

    if not args.skip_base_text:
        run_step("1/6 — Download base text and instruction data", [
            "download_training_data.py",
            "--wikitext-out", "data/raw/wikitext_train.txt",
            "--max-chars", str(args.max_base_chars),
            "--instructions-out", "data/orbit_instructions_full.jsonl",
        ])
    if not args.skip_capability_datasets:
        run_step("2/6 — Download mathematics, science, reading and coding data", [
            "download_capability_data.py",
            "--max-per-dataset", str(args.max_per_dataset),
        ])
    if not args.skip_live_data:
        run_step("3/6 — Collect live public data across seven domains", [
            "examples/real_data/build_data_lab.py",
            "--country", args.country, "--city", args.city,
        ])
    run_step("4/6 — Convert retrieved real records into sourced examples", [
        "examples/real_data/build_training_examples.py",
    ])
    run_step("5/6 — Prepare deduplicated base-language corpus", [
        "prepare_corpus.py",
    ])
    run_step("6/6 — Merge and balance instruction sources", [
        "merge_instruction_data.py",
    ])

    print("\nDATA PREPARATION COMPLETE")
    print("Base corpus: data/input.txt")
    print("Balanced instruction corpus: data/instructions_balanced.jsonl")
    print("Source manifests: data/capabilities/manifest.json and data/instructions_balanced.manifest.json")
    print("Next: review data/SOURCES.md, then train/resume FirstGPT and instruction-tune separately.")
    print("No model checkpoint was changed by this script.")


if __name__ == "__main__":
    main()
