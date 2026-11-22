#!/usr/bin/env python3
"""Stream clean, turn-aligned assistant examples from UltraChat.

Context is kept as a continuous suffix of complete messages. If the next
older message does not fit, selection stops rather than skipping it.
Resume restores hashes from the partial file to avoid duplicate examples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

DATASET = "HuggingFaceH4/ultrachat_200k"
SPLIT = "train_sft"
SOURCE_URL = "https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k"


def messages(row):
    out = []
    for item in row.get("messages") or row.get("conversation") or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or item.get("from") or "").lower()
        body = item.get("content", item.get("value"))
        role = {
            "human": "user",
            "user": "user",
            "assistant": "assistant",
            "gpt": "assistant",
        }.get(role)
        if role and isinstance(body, str) and body.strip():
            out.append((role, body.strip()))
    return out


def format_context(prior, max_context_chars, max_messages=12):
    """Keep the newest continuous suffix of complete messages."""
    selected = []
    used = 0

    for role, body in reversed(prior[-max_messages:]):
        prefix = "Owner: " if role == "user" else "Orbit: "
        line = prefix + body
        cost = len(line) + (1 if selected else 0)

        # Stop at the first message that cannot fit. Never skip over it.
        if used + cost > max_context_chars:
            break

        selected.append(line)
        used += cost

    selected.reverse()

    # A prompt should begin with a user message.
    while selected and not selected[0].startswith("Owner: "):
        selected.pop(0)

    # The final context message must be the user turn being answered.
    if not selected or not selected[-1].startswith("Owner: "):
        return ""

    return "\n".join(selected)


def examples(msgs, source_index, max_context_chars):
    prior = []
    for role, body in msgs:
        if role == "assistant" and 12 <= len(body) <= 1800:
            prompt = format_context(prior, max_context_chars)
            if prompt:
                response = "Orbit: " + body
                digest = hashlib.sha256(
                    (prompt + "\0" + response).encode("utf-8")
                ).hexdigest()
                yield {
                    "prompt": prompt,
                    "response": response,
                    "metadata": {
                        "dataset_id": "ultrachat_200k",
                        "dataset": DATASET,
                        "split": SPLIT,
                        "source_url": SOURCE_URL,
                        "license_declared_by_dataset_card": "MIT",
                        "source_record_index": source_index,
                        "turn_hash": digest,
                    },
                }
        prior.append((role, body))


def load_resume_state(partial):
    saved = 0
    skip = 0
    seen = set()
    if not partial.exists():
        return saved, skip, seen

    with partial.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                metadata = item.get("metadata") or {}
                saved += 1
                skip = max(
                    skip,
                    int(metadata.get("source_record_index", -1)) + 1,
                )
                digest = metadata.get("turn_hash")
                if not digest:
                    digest = hashlib.sha256(
                        (
                            item.get("prompt", "")
                            + "\0"
                            + item.get("response", "")
                        ).encode("utf-8")
                    ).hexdigest()
                seen.add(digest)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Cannot safely resume: invalid partial JSONL at "
                    f"line {line_number}: {exc}"
                ) from exc

    return saved, skip, seen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/instructions_conversations.jsonl"),
    )
    parser.add_argument("--max-context-chars", type=int, default=2500)
    args = parser.parse_args()

    if args.limit < 100 or args.max_context_chars < 200:
        parser.error("--limit must be >=100 and --max-context-chars must be >=200")

    from datasets import load_dataset

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")

    if args.output.exists() and args.output.stat().st_size:
        print(f"[UltraChat] already complete: {args.output}", flush=True)
        return

    saved, skip, seen = load_resume_state(partial)
    print(
        f"[UltraChat] streaming {DATASET}/{SPLIT}; resume {saved:,} turns "
        f"at source row {skip:,}; restored {len(seen):,} hashes",
        flush=True,
    )

    dataset = load_dataset(DATASET, split=SPLIT, streaming=True)
    count = saved
    started = time.monotonic()

    with partial.open("a" if saved else "w", encoding="utf-8") as output:
        for index, row in enumerate(dataset):
            if index < skip:
                continue
            if count >= args.limit:
                break

            for item in examples(messages(row), index, args.max_context_chars):
                digest = item["metadata"]["turn_hash"]
                if digest in seen:
                    continue

                seen.add(digest)
                output.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1

                if count % 100 == 0:
                    output.flush()
                    elapsed = max(time.monotonic() - started, 0.001)
                    print(
                        f"[UltraChat] {count:,}/{args.limit:,} turns | "
                        f"row {index:,} | {count / elapsed:.1f}/s",
                        flush=True,
                    )

                if count >= args.limit:
                    break

        output.flush()

    if count < 100:
        raise RuntimeError(
            f"UltraChat yielded only {count} turns; partial retained at {partial}"
        )

    partial.replace(args.output)
    manifest = {
        "dataset": DATASET,
        "split": SPLIT,
        "examples": count,
        "source_url": SOURCE_URL,
        "license_declared_by_dataset_card": "MIT",
        "context_truncation": (
            "newest continuous suffix of complete messages; "
            "stops at first oversized message"
        ),
        "resume_hashes_restored": len(seen),
        "warning": "Generated conversations are not independently fact-checked.",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[UltraChat] COMPLETE {count:,} turns -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
