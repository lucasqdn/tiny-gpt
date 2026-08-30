import torch
import time
import math
from model import TinyGPT
from token_data import TokenStream
from tokenizer import (
    decode as bpe_decode,
    encode as bpe_encode,
    load_tokenizer,
)
from values import d_model, dropout, max_seq_len, n_heads, n_layers
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

use_bf16 = (device == "cuda" and torch.cuda.is_bf16_supported())
print(f"Using BF16: {use_bf16}")

tokenizer_path = Path("tinystories_bpe_2048.json")
merges, vocab, eos_id, vocab_size = load_tokenizer(tokenizer_path)
print(f"BPE tokenizer loaded successfully. Vocabulary size: {vocab_size}")

torch.manual_seed(1337)
torch.cuda.manual_seed_all(1337)

batch_size = 64
sequence_length = 128

tokenized_directory = Path("data/TinyStories/tokenized")
train_stream = TokenStream(
    tokenized_directory / "train.bin",
    sequence_length,
)
validation_stream = TokenStream(
    tokenized_directory / "validation.bin",
    sequence_length,
)

steps_per_epoch = train_stream.number_of_blocks // batch_size
print(f"Training blocks: {train_stream.number_of_blocks:,}")
print(f"Steps per epoch: {steps_per_epoch:,}")

model = TinyGPT(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    dropout=dropout,
).to(device)

def encode(text):
    return bpe_encode(text, merges)

def decode(token_ids):
    return bpe_decode(token_ids, vocab, eos_id)

num_evaluation_batches = 20

fixed_training_ids = train_stream.create_epoch_order(seed=2024)[
    :num_evaluation_batches * batch_size
]
fixed_validation_ids = validation_stream.create_epoch_order(seed=2025)[
    :num_evaluation_batches * batch_size
]

fixed_training_batches = [
    train_stream.get_batch(
        fixed_training_ids[
            batch_index * batch_size:(batch_index + 1) * batch_size
        ],
        device,
    )
    for batch_index in range(num_evaluation_batches)
]

fixed_validation_batches = [
    validation_stream.get_batch(
        fixed_validation_ids[
            batch_index * batch_size:(batch_index + 1) * batch_size
        ],
        device,
    )
    for batch_index in range(num_evaluation_batches)
]

num_steps = steps_per_epoch
max_learning_rate = 3e-4
min_learning_rate = 3e-5
warmup_steps = 1000
run_name = "run9"
checkpoint_interval = 5000

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=max_learning_rate,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)

checkpoint_directory = Path("checkpoints")
checkpoint_directory.mkdir(exist_ok=True)
latest_checkpoint_path = checkpoint_directory / f"tiny_gpt_{run_name}_latest.pt"

# Leave this as None to start a new run. To resume, set it to a checkpoint:
# resume_checkpoint = Path("checkpoints/tiny_gpt_run9_latest.pt")
resume_checkpoint = None

if num_steps > steps_per_epoch:
    raise ValueError(
        f"num_steps {num_steps:,} exceeds one epoch of "
        f"{steps_per_epoch:,} steps"
    )

training_order = train_stream.create_epoch_order(seed=1337)

running_loss = 0
running_loss_steps = 0
initial_training_loss = None
final_training_loss = None
start_step = 0
previous_elapsed_time = 0.0

checkpoint_config = {
    "vocab_size": vocab_size,
    "max_seq_len": max_seq_len,
    "d_model": d_model,
    "n_heads": n_heads,
    "n_layers": n_layers,
    "dropout": dropout,
    "batch_size": batch_size,
    "sequence_length": sequence_length,
    "num_steps": num_steps,
    "max_learning_rate": max_learning_rate,
    "min_learning_rate": min_learning_rate,
    "warmup_steps": warmup_steps,
}


def get_learning_rate(step):
    # Linearly increase the learning rate during warmup.
    if step < warmup_steps:
        return max_learning_rate * (step + 1) / warmup_steps

    # Then smoothly decrease it from max_learning_rate to min_learning_rate.
    decay_steps = num_steps - warmup_steps
    decay_progress = (step - warmup_steps) / max(decay_steps - 1, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_learning_rate + cosine * (
        max_learning_rate - min_learning_rate
    )


def save_checkpoint(path, next_step, elapsed_training_time, metrics=None):
    checkpoint = {
        # next_step is the first step that has not been trained yet.
        "next_step": next_step,
        "num_steps": num_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "tokenizer_path": str(tokenizer_path),
        "eos_id": eos_id,
        "use_bf16": use_bf16,
        "elapsed_training_time": elapsed_training_time,
        "initial_training_loss": initial_training_loss,
        "final_training_loss": final_training_loss,
        "running_loss": running_loss,
        "running_loss_steps": running_loss_steps,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if device == "cuda" else None
        ),
        "config": checkpoint_config,
    }

    if metrics is not None:
        checkpoint.update(metrics)

    # Write to a temporary file first. replace() only changes the visible
    # checkpoint after torch.save has completed successfully.
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


if resume_checkpoint is not None:
    resume_checkpoint = Path(resume_checkpoint)
    checkpoint = torch.load(
        resume_checkpoint,
        map_location=device,
        weights_only=False,
    )

    if checkpoint.get("config") != checkpoint_config:
        raise ValueError(
            "The checkpoint configuration does not match the current "
            "model, batch size, or sequence length"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_step = checkpoint["next_step"]
    previous_elapsed_time = checkpoint.get("elapsed_training_time", 0.0)
    initial_training_loss = checkpoint.get("initial_training_loss")
    final_training_loss = checkpoint.get("final_training_loss")
    running_loss = checkpoint.get("running_loss", 0.0)
    running_loss_steps = checkpoint.get("running_loss_steps", 0)

    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if device == "cuda" and checkpoint.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["cuda_rng_state"]]
        )

    if start_step > num_steps:
        raise ValueError(
            f"Checkpoint starts at step {start_step:,}, but num_steps is "
            f"only {num_steps:,}"
        )

    print(f"Resuming from {resume_checkpoint} at step {start_step:,}")

# Puts the model in training mode
model.train()
if device == "cuda":
    torch.cuda.synchronize()
training_start_time = time.perf_counter()

for step in range(start_step, num_steps):
    learning_rate = get_learning_rate(step)
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate

    first_block = step * batch_size
    last_block = first_block + batch_size
    block_ids = training_order[first_block:last_block]

    inputs, targets = train_stream.get_batch(
        block_ids,
        device,
    )

    # Clear old gradients from previous training step
    optimizer.zero_grad(set_to_none=True)

    # Autocast uses BF16 for supported CUDA operations while keeping
    # numerically sensitive operations in FP32.
    with torch.autocast(
        device_type=device,
        dtype=torch.bfloat16,
        enabled=use_bf16,
    ):
        # logits: (B, T, vocab_size)
        # loss: scalar
        logits, loss = model(inputs, targets)

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite loss at step {step}: {loss.item()}"
        )

    current_loss = loss.item()
    if initial_training_loss is None:
        initial_training_loss = current_loss
    final_training_loss = current_loss

    # Calculates gradient for every parameter by tracing backwards
    loss.backward()

    # Prevent unusually large gradients
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    # Reads each parameter's gradient and changes the parameter
    optimizer.step()

    running_loss += current_loss
    running_loss_steps += 1

    if running_loss_steps == 10:
        average_loss = running_loss / running_loss_steps
        print(
            f"step {step + 1}: loss {average_loss:.4f}, "
            f"lr {learning_rate:.2e}"
        )
        running_loss = 0.0
        running_loss_steps = 0

    if (step + 1) % checkpoint_interval == 0:
        if device == "cuda":
            torch.cuda.synchronize()
        checkpoint_elapsed_time = (
            previous_elapsed_time
            + time.perf_counter()
            - training_start_time
        )
        save_checkpoint(
            latest_checkpoint_path,
            next_step=step + 1,
            elapsed_training_time=checkpoint_elapsed_time,
        )
        print(f"Saved checkpoint to {latest_checkpoint_path}")

if device == "cuda":
    torch.cuda.synchronize()
elapsed_time = (
    previous_elapsed_time
    + time.perf_counter()
    - training_start_time
)

total_tokens_trained = num_steps * batch_size * sequence_length
tokens_per_second = total_tokens_trained / elapsed_time

# Validation measures performance without calculating gradients or updating weights.

@torch.no_grad()
def average_loss(model, batches):
    # Check if the model is training, if yes, then it will return True
    was_training = model.training
    model.eval()

    losses = []

    for inputs, targets in batches:
        with torch.autocast(
            device_type=device,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            _, loss = model(inputs, targets)

        losses.append(loss.item())

    if was_training:
        model.train()

    return sum(losses) / len(losses)

average_training_loss = average_loss(model, fixed_training_batches)
average_validation_loss = average_loss(model, fixed_validation_batches)

print("\nRun summary")
print(f"Initial training loss: {initial_training_loss:.4f}")
print(f"Final training loss: {final_training_loss:.4f}")
print(f"Average training loss: {average_training_loss:.4f}")
print(f"Average validation loss: {average_validation_loss:.4f}")
print(f"Elapsed training time: {elapsed_time:.2f} seconds")
print(f"Tokens per second: {tokens_per_second:,.0f}")
print(f"Total tokens trained: {total_tokens_trained:,}")

model.eval()

prompt = "Once upon a time"

prompt_ids = torch.tensor(
    [encode(prompt)],
    dtype=torch.long,
    device=device,
)

with torch.autocast(
    device_type=device,
    dtype=torch.bfloat16,
    enabled=use_bf16,
):
    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=300,
        temperature=0.8,
        eos_id=eos_id,
    )

print("\nGenerated sample:")
print(decode(generated_ids[0].tolist()))

stopped_at_eos = generated_ids[0, -1].item() == eos_id
generated_tokens = generated_ids.shape[1] - prompt_ids.shape[1]

print(f"Stopped at EOS: {stopped_at_eos}")
print(f"Generated tokens: {generated_tokens}")


checkpoint_path = (
    checkpoint_directory / f"tiny_gpt_{run_name}_step_{num_steps}.pt"
)

save_checkpoint(
    checkpoint_path,
    next_step=num_steps,
    elapsed_training_time=elapsed_time,
    metrics={
        "average_training_loss": average_training_loss,
        "average_validation_loss": average_validation_loss,
    },
)
print(f"Saved checkpoint to {checkpoint_path}")
