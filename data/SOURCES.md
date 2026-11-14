# Training dataset sources and attribution

The optional download_training_data.py script downloads these datasets directly into the workspace. The notebook uses WikiText-103 and caps the local training text at five million characters by default to provide more signal than the earlier WikiText-2 one-million-character starter. No user upload or Google Drive mount is required.

## WikiText

- Dataset: https://huggingface.co/datasets/Salesforce/wikitext
- Default configuration: wikitext-103-raw-v1
- Purpose: base next-token language-model pretraining.
- License metadata includes CC BY-SA 3.0 and GFDL; the dataset card's licensing paragraph also refers to CC BY-SA 4.0. Check the current card and upstream source terms before redistributing trained artifacts.
- The WikiText dataset card describes over 100 million tokens across WikiText-103 and smaller WikiText-2 variants.

## Databricks Dolly 15K

- Dataset: https://huggingface.co/datasets/databricks/databricks-dolly-15k
- Purpose: supervised instruction-following examples, including brainstorming, classification, question answering, generation, information extraction, and summarisation.
- Declared license: CC BY-SA 3.0 Unported.
- The download script includes the instruction, optional context, and response fields and appends Orbit-specific examples in data/orbit_instructions.example.jsonl.
- Dataset card and attribution: Databricks, Inc. (2023), Dolly 15K.

These datasets improve the breadth of the training signal but do not guarantee factual accuracy or high assistant quality. WikiText is primarily English Wikipedia text; Dolly is a relatively small instruction dataset. The small FirstGPT architecture and training budget remain important limitations.

## Capability-specific instruction datasets

`download_capability_data.py` downloads only the **training split** of the selected sources and converts them into FirstAI's `prompt`/`response` JSONL format. Every row retains dataset ID, original split, declared license, source URL, purpose and source row index in a metadata object.

| Capability | Dataset | Declared license | Source |
|---|---|---|---|
| Mathematics | GSM8K (`openai/gsm8k`) | MIT | https://huggingface.co/datasets/openai/gsm8k |
| School science | AI2 ARC Challenge (`allenai/ai2_arc`) | CC BY-SA 4.0 | https://huggingface.co/datasets/allenai/ai2_arc |
| Reading comprehension | SQuAD (`rajpurkar/squad`) | CC BY-SA 4.0 | https://huggingface.co/datasets/rajpurkar/squad |
| Python coding | MBPP sanitized (`google-research-datasets/mbpp`) | CC BY 4.0 | https://huggingface.co/datasets/google-research-datasets/mbpp |

The downloader deliberately does not fetch test/validation splits. Keep a separate held-out evaluation set; do not train on benchmark answers and then report those same questions as evaluation.

Run:

```bash
pip install datasets
python download_capability_data.py --max-per-dataset 3000
```

Outputs:
- `data/capabilities/*_train.jsonl` — per-source records.
- `data/instructions_capabilities.jsonl` — combined instruction data.
- `data/capabilities/manifest.json` — provenance manifest.

Instruction-tune an existing checkpoint with:

```bash
python finetune_instructions.py --checkpoint checkpoints/firstgpt.best.pt --data data/instructions_capabilities.jsonl --out checkpoints/firstgpt-capabilities.pt
```

This is a small, capped starter mix, not a claim of professional-grade or comprehensive knowledge. Review each current dataset card and comply with the license's attribution and share-alike terms before redistributing data or model artifacts.

## Live knowledge and reference retrieval

Orbit's `knowledge_search` tool uses Wikipedia REST for general encyclopedic discovery, OpenAlex and arXiv for scholarly discovery, and Stack Exchange for programming question discovery. Results carry retrieval timestamps, source URLs, license notes and caveats. These sources return discovery metadata/excerpts; they do not automatically download entire articles or prove that a claim is correct.

Orbit's existing `live_data` tool remains separate and handles structured public datasets such as weather/air-quality forecasts, USGS earthquakes, reference exchange rates, country profiles, World Bank indicators and NASA natural events.

Image-reference search uses Wikimedia Commons as a keyless strict-safe fallback when Brave image search is not configured. It returns per-file license and attribution metadata where available. Verify the original file page before reusing an image.



## Checked-in real-data seed examples

`data/raw/real_world_examples.txt` and `data/raw/capability_seed_examples.txt` are small source-attributed starter samples. Their structured provenance companions are `examples/real_data/seed_examples.jsonl` and `examples/real_data/capability_seed_examples.jsonl`. These are included by `prepare_corpus.py`; they are not a replacement for full source datasets.

## One-command corpus preparation

Run `pip install datasets` followed by `python build_all_data.py` to download the base WikiText/Dolly datasets, the capability-specific instruction datasets, collect the seven-domain public data lab, generate source-grounded examples, prepare `data/input.txt`, and merge a balanced `data/instructions_balanced.jsonl` instruction set. The pipeline stops on a failed stage and preserves earlier outputs. It does not train a checkpoint.

Use `python finetune_instructions.py --checkpoint checkpoints/firstgpt.best.pt --data data/instructions_balanced.jsonl --out checkpoints/firstgpt-capabilities.pt` after reviewing the source manifest and training mix.


## Larger, resumable training profile

The default Colab profile requests up to 30 million WikiText characters, up to 10,000 rows per capability training split, and up to 10,000 instruction examples per dataset source after deduplication/balancing. Collection caps are configurable because Colab disk/RAM and GPU session duration vary. Only training splits are downloaded for these instruction datasets; reserve separate benchmarks for evaluation.

Base pretraining checkpoints every 250 optimizer steps and resumes only with the same corpus hash. SFT saves latest and best-validation checkpoints separately and resumes only when the instruction JSONL hash matches. A changed dataset requires a fresh output checkpoint or explicit restart; it is never silently mixed into an old run.

The Colab T4 profile defaults to 10,000 base steps and 5,000 SFT steps with a 6-layer, width-256, 1,024-token model. This remains a small experimental GPT trained from scratch, not a competitive foundation model. More steps alone do not replace architecture, data scale, data quality or evaluation.

Termux uses a separate compact CPU training profile to reduce heat and RAM pressure. It also collects a fresh public-data snapshot locally and derives its own practical instruction examples. It is not the same model as the Colab checkpoint.
