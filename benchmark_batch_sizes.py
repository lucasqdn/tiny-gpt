import argparse
import gc
import time
from pathlib import Path

import torch

from model import TinyGPT
from token_data import TokenStream
from tokenizer import load_tokenizer
from values import d_model, dropout, max_seq_len, n_heads, n_layers


TOKENIZER_PATH = Path("tinystories_bpe_2048.json")
TRAIN_TOKENS_PATH = Path("data/TinyStories/tokenized/train.bin")
SEQUENCE_LENGTH = 128
SEED = 1337


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark TinyGPT training throughput by batch size."
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128],
    )
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--measure-steps", type=int, default=250)
    return parser.parse_args()


def run_training_step(model, optimizer, inputs, targets):
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        _, loss = model(inputs, targets)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.detach()


def benchmark_batch_size(
    batch_size,
    stream,
    epoch_order,
    vocab_size,
    warmup_steps,
    measure_steps,
):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model = TinyGPT(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    ).to("cuda")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    model.train()
    total_steps = warmup_steps + measure_steps
    last_loss = None

    for step in range(total_steps):
        if step == warmup_steps:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            measurement_start = time.perf_counter()

        first_block = step * batch_size
        last_block = first_block + batch_size
        block_ids = epoch_order[first_block:last_block]

        inputs, targets = stream.get_batch(block_ids, "cuda")

        last_loss = run_training_step(
            model,
            optimizer,
            inputs,
            targets,
        )

    torch.cuda.synchronize()
    elapsed_time = time.perf_counter() - measurement_start

    if not torch.isfinite(last_loss):
        raise RuntimeError(
            f"Non-finite loss for batch size {batch_size}: "
            f"{last_loss.item()}"
        )

    measured_tokens = (
        measure_steps
        * batch_size
        * SEQUENCE_LENGTH
    )
    tokens_per_second = measured_tokens / elapsed_time
    peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3

    return {
        "batch_size": batch_size,
        "tokens_per_second": tokens_per_second,
        "peak_memory_gb": peak_memory_gb,
        "loss": last_loss.item(),
    }


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16")

    if args.warmup_steps < 1 or args.measure_steps < 1:
        raise ValueError("Warmup and measurement steps must be positive")

    _, _, _, vocab_size = load_tokenizer(TOKENIZER_PATH)
    stream = TokenStream(TRAIN_TOKENS_PATH, SEQUENCE_LENGTH)

    largest_batch_size = max(args.batch_sizes)
    required_blocks = (
        (args.warmup_steps + args.measure_steps)
        * largest_batch_size
    )

    if required_blocks > stream.number_of_blocks:
        raise ValueError("The benchmark requests more blocks than exist")

    epoch_order = stream.create_epoch_order(seed=SEED)
    results = []

    print("Using CUDA BF16")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Measured steps: {args.measure_steps}")

    for batch_size in args.batch_sizes:
        print(f"\nBenchmarking batch size {batch_size}...")

        try:
            result = benchmark_batch_size(
                batch_size=batch_size,
                stream=stream,
                epoch_order=epoch_order,
                vocab_size=vocab_size,
                warmup_steps=args.warmup_steps,
                measure_steps=args.measure_steps,
            )
        except torch.OutOfMemoryError:
            print(f"Batch size {batch_size}: CUDA out of memory")
            result = None

        if result is not None:
            results.append(result)
            print(
                f"Batch size {batch_size}: "
                f"{result['tokens_per_second']:,.0f} tokens/s, "
                f"{result['peak_memory_gb']:.2f} GB peak, "
                f"loss {result['loss']:.4f}"
            )

        gc.collect()
        torch.cuda.empty_cache()

    if not results:
        raise RuntimeError("Every tested batch size ran out of memory")

    best_result = max(
        results,
        key=lambda result: result["tokens_per_second"],
    )

    print("\nSummary")
    print("Batch | Tokens/s | Peak memory | Final loss")

    for result in results:
        print(
            f"{result['batch_size']:>5} | "
            f"{result['tokens_per_second']:>8,.0f} | "
            f"{result['peak_memory_gb']:>8.2f} GB | "
            f"{result['loss']:.4f}"
        )

    print(
        f"\nBest throughput: batch size "
        f"{best_result['batch_size']} at "
        f"{best_result['tokens_per_second']:,.0f} tokens/s"
    )


if __name__ == "__main__":
    main()
