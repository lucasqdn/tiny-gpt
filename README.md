# TinyGPT

TinyGPT is a 15-million-parameter decoder-only transformer trained on the
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.
The project implements causal self-attention, transformer blocks, text
generation, and a byte-pair encoding (BPE) tokenizer directly in PyTorch as a
learning exercise.

The final model was trained for one full-batch pass over essentially all 465
million BPE tokens on an NVIDIA GeForce RTX 5070 Ti. It reached a validation
loss of **1.7602**, generated complete short stories, and stopped at the
end-of-story token in **20/20** fixed evaluation samples.

## What is implemented

- Byte-level BPE training, encoding, decoding, and tokenizer serialization
- Heap-based BPE encoding with linked-list-style token indices
- Token and learned positional embeddings
- Vectorized multi-head causal self-attention
- Pre-normalization transformer blocks with residual connections
- GELU feed-forward networks
- Tied input-embedding and language-model-head weights
- Autoregressive generation with temperature and EOS stopping
- Memory-mapped, pretokenized training and validation streams
- BF16 mixed-precision training on supported CUDA GPUs
- AdamW, gradient clipping, linear warmup, and cosine learning-rate decay
- Fixed training and validation batches for comparable evaluation
- Atomic checkpoint saving and resume support with RNG-state restoration
- Batch-size throughput and peak-memory benchmarking

This project does not use `nn.MultiheadAttention` or `nn.Transformer`. It does
use standard PyTorch components such as `nn.Linear`, `nn.Embedding`,
`nn.LayerNorm`, and `torch.optim.AdamW`.

## Architecture

| Setting | Value |
|---|---:|
| Parameters | 15,004,416 |
| Vocabulary | 2,048 byte-level BPE tokens |
| Context length | 128 tokens |
| Embedding width (`d_model`) | 384 |
| Transformer layers | 8 |
| Attention heads | 6 |
| Dimension per head | 64 |
| Feed-forward width | 1,536 |
| Dropout | 0.0 |

Each block uses the pre-norm layout:

```text
x = x + causal_self_attention(layer_norm(x))
x = x + feed_forward(layer_norm(x))
```

The attention implementation projects `(B, T, C)` into queries, keys, and
values, rearranges them to `(B, H, T, D)`, applies a cached causal mask, and
combines the heads back into `(B, T, C)`. The final language-model head maps
each position to 2,048 next-token logits.

## Final training result

Run 9 trained from scratch for one shuffled, full-batch pass over the dataset:

| Metric | Result |
|---|---:|
| Training steps | 56,820 |
| Tokens trained | 465,469,440 |
| Initial training loss | 7.6972 |
| Final training loss | 1.7637 |
| Average training loss | 1.7670 |
| Average validation loss | 1.7602 |
| Training time | 1,240.95 seconds |
| Throughput | 375,092 tokens/second |

Generation evaluation used 20 reproducible samples with a maximum of 500 new
tokens:

| Metric | Result |
|---|---:|
| EOS stops | 20/20 |
| EOS stop rate | 100% |
| Average generated length | 134.9 BPE tokens |
| Average generated text | 599.9 characters |

Example from the final run:

> Once upon a time, a little girl named Lily went for a walk with her mom. They
> saw a big tree and Lily said, "Look mommy, the tree is so pretty!" Her mom
> said, "Yes, it's a rich tree." [...] And they lived happily ever after. The
> end.

The full experiment history and longer generated samples are in
[`LOG.md`](LOG.md).

## Setup

Create and activate a virtual environment, then install the pinned packages:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU is recommended. Training falls back to CPU, but the batch-size
benchmark specifically requires CUDA and BF16 support.

## Prepare the data

Download TinyStories and save its splits locally:

```bash
python data.py
```

Train the 2,048-token BPE tokenizer:

```bash
python tokenizer.py
```

The current tokenizer is trained on the first 1,000 training stories and saves
its merge rules to `tinystories_bpe_2048.json`. Token ID 2,047 is reserved for
EOS.

Pretokenize the complete dataset into contiguous `uint16` streams:

```bash
python pretokenize.py
```

This produces:

```text
data/TinyStories/tokenized/train.bin
data/TinyStories/tokenized/validation.bin
data/TinyStories/tokenized/metadata.json
```

The binary streams are memory-mapped during training, so the full dataset does
not need to be loaded into RAM.

## Find a suitable batch size

The benchmark runs short BF16 training trials and reports throughput and peak
allocated GPU memory:

```bash
python benchmark_batch_sizes.py
```

Custom batch sizes and trial lengths can be supplied:

```bash
python benchmark_batch_sizes.py \
  --batch-sizes 16 32 64 128 \
  --warmup-steps 50 \
  --measure-steps 250
```

Batch size 64 produced the best measured throughput on the RTX 5070 Ti.

## Train

The main hyperparameters are in `values.py` and near the top of `train.py`.
After preparing the dataset, start training with:

```bash
python train.py
```

The current configuration trains one complete epoch using:

- Batch size 64
- Sequence length 128
- 1,000 linear warmup steps
- Cosine decay from `3e-4` to `3e-5`
- A checkpoint every 5,000 steps

Periodic checkpoints overwrite `checkpoints/tiny_gpt_run9_latest.pt` so they do
not accumulate indefinitely. A completed run also writes a step-numbered final
checkpoint.

### Resume an interrupted run

Set this value in `train.py`:

```python
resume_checkpoint = Path("checkpoints/tiny_gpt_run9_latest.pt")
```

Then run `python train.py` again. The checkpoint restores the model, AdamW
state, RNG state, next training step, elapsed time, and loss bookkeeping. Keep
the model, batch, sequence, and scheduler settings unchanged when resuming.

Set `resume_checkpoint` back to `None` before beginning a new run.

> **Current limitation:** the training order and guard in `train.py` are set up
> for one epoch. Multi-epoch training needs a new shuffled order for each epoch;
> simply multiplying `num_steps` will not work correctly.

## Evaluate generation

`evaluate.py` loads the final Run 9 checkpoint, generates 20 deterministic
samples, and reports EOS behavior and generated lengths:

```bash
python evaluate.py
```

Change `checkpoint_path` inside `evaluate.py` when evaluating a different run.
Sampling currently uses temperature `0.8` and a limit of 500 new tokens.

## Project layout

| File | Purpose |
|---|---|
| `model.py` | Attention, feed-forward network, transformer block, TinyGPT, and generation |
| `tokenizer.py` | BPE training, optimized encoding, decoding, saving, and loading |
| `data.py` | Downloads and saves TinyStories |
| `pretokenize.py` | Converts dataset splits into packed `uint16` token streams |
| `token_data.py` | Memory-mapped token stream and deterministic batch construction |
| `train.py` | BF16 training, evaluation, scheduling, generation, and checkpoints |
| `evaluate.py` | Reproducible generation and EOS evaluation |
| `benchmark_batch_sizes.py` | CUDA throughput and peak-memory benchmark |
| `values.py` | Core model dimensions |
| `LOG.md` | Results and observations from Runs 1–9 |
| `notebooks/` | Exploratory and tutorial notebooks from development |
| `archive/` | Preserved tokenizer and dataset artifacts from earlier experiments |

## Progression

The project began with character-level tokens and an explicit Python module per
attention head. Later runs added EOS-aware packed data, custom BPE, vectorized
QKV attention, pretokenization, memory mapping, BF16, larger batches, scheduling,
and resumable checkpoints.

The largest throughput improvements were:

| Run | Main change | Throughput |
|---|---|---:|
| 6 | BPE baseline | 33,847 tokens/s |
| 7 | Vectorized attention | 68,634 tokens/s |
| 8 | Pretokenization, BF16, batch size 64 | 387,013 tokens/s |
| 9 | Full-dataset training with scheduling | 375,092 tokens/s |

The final pipeline is about 11 times faster in raw BPE-token throughput than the
initial BPE training loop while training on substantially more source text per
token than the earlier character-level model.

## Known limitations

- The context window is only 128 BPE tokens, limiting long-range story
  consistency.
- The tokenizer was trained on 1,000 stories rather than the complete corpus.
- Generation recomputes the context every step; there is no KV cache.
- Sampling supports temperature but not top-k or top-p filtering.
- Training is single-process and single-GPU.
- Generated stories are structurally coherent but can still contain repetition
  and locally incorrect meanings.
