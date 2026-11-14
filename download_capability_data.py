#!/usr/bin/env python3
"""Download licensed capability datasets into auditable instruction JSONL.

Only training splits are downloaded. Evaluation/test splits are deliberately
excluded. Each output row carries source, split, license, dataset revision,
and original record ID where available.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SOURCES = {
    "gsm8k": {
        "dataset": "openai/gsm8k", "config": "main", "split": "train",
        "license": "MIT", "url": "https://huggingface.co/datasets/openai/gsm8k",
        "purpose": "mathematical word-problem reasoning",
    },
    "arc": {
        "dataset": "allenai/ai2_arc", "config": "ARC-Challenge", "split": "train",
        "license": "CC-BY-SA-4.0", "url": "https://huggingface.co/datasets/allenai/ai2_arc",
        "purpose": "school science question answering",
    },
    "squad": {
        "dataset": "rajpurkar/squad", "config": "plain_text", "split": "train",
        "license": "CC-BY-SA-4.0", "url": "https://huggingface.co/datasets/rajpurkar/squad",
        "purpose": "reading comprehension grounded in supplied passages",
    },
    "mbpp": {
        "dataset": "google-research-datasets/mbpp", "config": "sanitized", "split": "train",
        "license": "CC-BY-4.0", "url": "https://huggingface.co/datasets/google-research-datasets/mbpp",
        "purpose": "Python programming tasks with reference solutions and tests",
    },
}

def _load(source):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the optional downloader dependency: pip install datasets") from exc
    return load_dataset(source["dataset"], source["config"], split=source["split"])


def _convert(name, row, index, source):
    if name == "gsm8k":
        prompt = "Solve this mathematics word problem. Show a clear, checkable solution.\n\n" + str(row.get("question", "")).strip()
        response = str(row.get("answer", "")).strip()
        if not response:
            return None
    elif name == "arc":
        question = str(row.get("question", "")).strip()
        choices = row.get("choices") or {}
        labels = choices.get("label") or []
        texts = choices.get("text") or []
        options = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
        answer_key = str(row.get("answerKey", "")).strip()
        answer_text = next((text for label, text in zip(labels, texts) if str(label) == answer_key), "")
        if not question or not answer_key or not answer_text:
            return None
        prompt = "Answer this science question using the choices. Explain briefly why the selected answer fits.\n\n" + question + "\n" + options
        response = f"{answer_key}. {answer_text}"
    elif name == "squad":
        context = str(row.get("context", "")).strip()
        question = str(row.get("question", "")).strip()
        answers = (row.get("answers") or {}).get("text") or []
        answer = str(answers[0]).strip() if answers else ""
        if not context or not question or not answer:
            return None
        prompt = "Answer only from the passage. If the passage does not support an answer, say so.\n\nPassage:\n" + context + "\n\nQuestion: " + question
        response = answer
    elif name == "mbpp":
        task = str(row.get("text", "")).strip()
        code = str(row.get("code", "")).strip()
        if not task or not code:
            return None
        tests = row.get("test_list") or []
        prompt = "Write a Python function for this programming task. Return code only, and satisfy the listed tests.\n\nTask: " + task
        if tests:
            prompt += "\n\nTests:\n" + "\n".join(str(test) for test in tests[:8])
        response = code
    else:
        return None
    return {
        "prompt": prompt,
        "response": response,
        "metadata": {
            "dataset_id": name,
            "dataset": source["dataset"],
            "config": source["config"],
            "split": source["split"],
            "license_declared_by_dataset_card": source["license"],
            "source_url": source["url"],
            "purpose": source["purpose"],
            "source_record_index": index,
        },
    }


def download_one(name, limit, output_dir):
    source = SOURCES[name]
    dataset = _load(source)
    target = output_dir / f"{name}_train.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(dataset):
            if written >= limit:
                break
            item = _convert(name, row, index, source)
            if item is None:
                continue
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1
    if written < 4:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"{name} produced only {written} usable examples; refusing an empty/weak output.")
    return {"dataset_id": name, "dataset": source["dataset"], "license": source["license"],
            "split": source["split"], "examples": written, "path": str(target)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=tuple(SOURCES), default=list(SOURCES))
    parser.add_argument("--max-per-dataset", type=int, default=3000)
    parser.add_argument("--output-dir", type=Path, default=Path("data/capabilities"))
    parser.add_argument("--combined", type=Path, default=Path("data/instructions_capabilities.jsonl"))
    args = parser.parse_args()
    if args.max_per_dataset < 4:
        parser.error("--max-per-dataset must be at least 4")
    output_dir = args.output_dir.expanduser().resolve()
    results = [download_one(name, args.max_per_dataset, output_dir) for name in args.datasets]
    combined = args.combined.expanduser().resolve()
    combined.parent.mkdir(parents=True, exist_ok=True)
    with combined.open("w", encoding="utf-8") as target:
        for result in results:
            source_path = Path(result["path"])
            with source_path.open(encoding="utf-8") as source:
                for line in source:
                    target.write(line)
    manifest = {
        "schema_version": 1,
        "created_by": "FirstAI capability dataset downloader",
        "warning": "Declared licenses are recorded from dataset metadata; review each current dataset card and comply with attribution/share-alike terms before redistribution.",
        "evaluation_split_policy": "Only source training splits are downloaded; do not use these rows as an evaluation set.",
        "datasets": results,
        "combined_instruction_file": str(combined),
        "examples_total": sum(item["examples"] for item in results),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
