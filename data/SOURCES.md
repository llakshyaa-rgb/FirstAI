# Training dataset sources and attribution

The optional download_training_data.py script downloads these datasets directly into the workspace. No user upload or Google Drive mount is required.

## WikiText

- Dataset: https://huggingface.co/datasets/Salesforce/wikitext
- Default configuration: wikitext-2-raw-v1
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