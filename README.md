# FirstAI

An experimental decoder-only language model built from random initialization.
No pretrained model or hosted language-model API is used.

## What is here

- Multi-head causal self-attention
- Residual Transformer blocks with pre-layer normalization
- Learned token and position embeddings
- Character tokenizer (deliberately simple and inspectable)
- Next-token training objective
- AdamW optimization, gradient clipping, validation loss and checkpoints
- Autoregressive text generation

PyTorch supplies tensor operations, automatic differentiation and device
execution. The model architecture, tokenizer and training loop are our own.
This means the model is trained from scratch; it does not mean we have
reimplemented low-level GPU kernels or automatic differentiation.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Put a UTF-8 corpus in `data/input.txt`, then:

```bash
python train.py --steps 2000
python generate.py --prompt "Your prompt"
```

The trainer automatically selects CUDA when available, otherwise CPU. For a
free Colab GPU session, clone this repository into the notebook, upload or
mount your corpus, install requirements, and run the same commands.

## Important

The included model is a learning/research baseline, not a capable assistant.
A small corpus and small model will produce limited, repetitive text. Better
data, tokenization, evaluation, scaling and safety work are separate milestones.
