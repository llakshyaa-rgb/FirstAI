# FirstAI

An experimental decoder-only language model trained from random initialization.
No pretrained weights or hosted language-model API are used.

## Architecture

- Multi-head causal self-attention with a future-token mask
- Pre-layer-normalized residual Transformer blocks
- Learned token and positional embeddings
- GELU feed-forward networks and tied input/output embeddings
- Next-token cross-entropy objective
- AdamW, gradient clipping, train/validation loss and perplexity
- Best-checkpoint saving and resumable optimizer state
- Byte-level BPE tokenizer implemented in this repository

PyTorch supplies tensor kernels, automatic differentiation and device execution.
The architecture, tokenizer and training loop are implemented here. This is a
from-scratch model, not a reimplementation of GPU kernels or autograd.

## Prepare a corpus

Place multiple UTF-8 `.txt` documents in `data/raw/`, then run:

```bash
python prepare_corpus.py
```

This normalizes Unicode, removes exact duplicate paragraphs and writes
`data/input.txt`. Only use material you have permission to train on.

## Train and generate

Install:

```bash
pip install -r requirements.txt
```

Then:

```bash
python train.py --steps 2000
python generate.py --prompt "Your prompt"
```

Resume from the saved checkpoint using the same corpus and model settings:

```bash
python train.py --resume --steps 2000
```

The trainer selects CUDA when available and otherwise uses CPU. The included
`FirstAI_Colab.ipynb` notebook guides you through uploading a corpus and
training on a Colab GPU when one is available.

## Tests

```bash
python -m unittest discover -s tests
```

## Important limitations

This is a research/learning baseline, not yet a capable assistant. A small
model and small corpus will produce limited output. Corpus curation, robust
evaluation, scaling, instruction tuning and safety are later milestones.
