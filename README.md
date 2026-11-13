# FirstAI

FirstAI is a decoder-only language model trained from random initialization.
It does not load pretrained weights or call a hosted language-model API.

## Implemented technical foundation

- Byte-level BPE tokenizer implemented in this repository
- Learned token and positional embeddings
- Multi-head causal self-attention with a future-token mask
- Pre-layer-normalized residual Transformer blocks
- GELU feed-forward networks and tied input/output embeddings
- Next-token cross-entropy objective and AdamW optimization
- Gradient clipping, gradient accumulation, optional CUDA mixed precision, train/validation loss and validation perplexity
- Best-model checkpoints, periodic resumable checkpoints, corpus integrity checks and RNG-state restoration
- Text generation from a saved checkpoint

PyTorch supplies tensor operations, automatic differentiation and device execution.
The model architecture, tokenizer, corpus preparation and training loop are custom.
This is a from-scratch model, not a fine-tune of an existing language model.

## Prepare a corpus

Place UTF-8 `.txt` documents that you have permission to use in `data/raw/`.
Then run:

```bash
python prepare_corpus.py
```

The preparation script normalizes Unicode, removes exact duplicate paragraphs
and writes `data/input.txt`. It does not automatically verify factual quality,
remove every form of boilerplate, or grant rights to source material.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --steps 2000
```

Useful options include `--steps`, `--block-size`, `--layers`, `--heads`,
`--width`, `--batch-size`, `--eval-interval` and `--checkpoint-interval`.
The trainer automatically uses CUDA when available and otherwise CPU.

Resume from the same corpus and checkpoint:

```bash
python train.py --resume --steps 2000
```

The checkpoint stores a SHA-256 fingerprint of the corpus and rejects a changed
corpus rather than silently resuming incompatible training. Saved model
architecture and optimizer state are restored from the checkpoint.

## Evaluate

Evaluate a saved checkpoint against the same corpus used for training:

```bash
python evaluate.py --checkpoint checkpoints/firstgpt.best.pt --data data/input.txt
```

The evaluator checks the corpus fingerprint before reporting train and held-out
validation loss/perplexity.

## Pipeline smoke test

Run a temporary synthetic-data test of training, checkpoint creation, resume
and generation without uploading a real corpus:

```bash
python scripts/smoke_test.py
```

## Generate

```bash
python generate.py --checkpoint checkpoints/firstgpt.best.pt --prompt "The " --tokens 200
```

## Tests

```bash
python -m unittest discover -s tests
```

Tests cover BPE UTF-8 round-tripping, unseen characters, corpus deduplication,
next-token batch alignment, model tensor shapes, causal masking, gradient flow
and generation.

## Colab

Open [FirstAI_Colab.ipynb](https://colab.research.google.com/github/llakshyaa-rgb/FirstAI/blob/main/FirstAI_Colab.ipynb).
The notebook uses the PyTorch installation already provided by Colab, runs unit tests and an end-to-end synthetic-data smoke test, and
automatically falls back to CPU if CUDA is unavailable. A free GPU is not
guaranteed. Download checkpoints before the temporary runtime disconnects.

## Scope and limitations

This completes the code foundation for training and running a small GPT-style
language model. It does not mean the model has been trained or has learned
useful language yet. Quality depends on corpus volume and quality, compute,
model scale and subsequent training. Personality, conversational behaviour,
instruction tuning, memory, tool use and assistant policies are deliberately
outside this milestone.
