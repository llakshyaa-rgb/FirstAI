"""Build a clean plain-text corpus from a folder of UTF-8 .txt documents."""
import argparse
from pathlib import Path
import unicodedata


def collect_documents(folder, min_paragraph_chars=1):
    paths = sorted(Path(folder).rglob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No .txt files found under {folder}")

    seen = set()
    documents = []
    stats = {"files": 0, "paragraphs_seen": 0, "duplicates": 0, "kept": 0}

    for path in paths:
        raw = path.read_text(encoding="utf-8-sig")
        raw = unicodedata.normalize("NFC", raw).replace("\r\n", "\n").replace("\r", "\n")
        stats["files"] += 1
        paragraphs = [part.strip() for part in raw.split("\n\n")]
        for paragraph in paragraphs:
            paragraph = "\n".join(line.rstrip() for line in paragraph.splitlines()).strip()
            if not paragraph:
                continue
            stats["paragraphs_seen"] += 1
            if len(paragraph) < min_paragraph_chars:
                continue
            if paragraph in seen:
                stats["duplicates"] += 1
                continue
            seen.add(paragraph)
            documents.append(paragraph)
            stats["kept"] += 1

    return "\n\n".join(documents) + "\n", stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--output", default="data/input.txt")
    parser.add_argument("--min-paragraph-chars", type=int, default=1)
    args = parser.parse_args()

    corpus, stats = collect_documents(args.input_dir, args.min_paragraph_chars)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(corpus, encoding="utf-8")
    print(f"Source files: {stats['files']}")
    print(f"Paragraphs found: {stats['paragraphs_seen']}")
    print(f"Exact duplicates removed: {stats['duplicates']}")
    print(f"Paragraphs kept: {stats['kept']}")
    print(f"Output: {output} ({len(corpus):,} characters)")


if __name__ == "__main__":
    main()
