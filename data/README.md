# Training data

Put a UTF-8 plain-text corpus at `data/input.txt`. It is intentionally not
included in Git so the repository does not redistribute somebody else's
content or accidentally publish private text.

## Corpus quality matters

- Use text you have permission to use.
- Prefer clean, well-formed, varied documents over repeated filler.
- Remove navigation menus, duplicated pages, broken encoding and boilerplate.
- The trainer reserves the final 10% of the supplied text for validation.
- A tiny corpus checks that the pipeline runs; it does not make a useful
  general-purpose assistant.

## Format

Plain UTF-8 text. Separate documents with blank lines. The byte-level BPE
tokenizer can represent any UTF-8 character, including unseen characters.
