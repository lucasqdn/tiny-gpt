import torch
import time
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

tokenizer_path = Path("tinystories_bpe_2048.json")
merges, vocab, eos_id, vocab_size = load_tokenizer(tokenizer_path)
print(f"BPE tokenizer loaded successfully. Vocabulary size: {vocab_size}")

torch.manual_seed(1337)
torch.cuda.manual_seed_all(1337)

batch_size = 8
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

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)

num_steps = 5000

if num_steps > steps_per_epoch:
    raise ValueError(
        f"num_steps {num_steps:,} exceeds one epoch of "
        f"{steps_per_epoch:,} steps"
    )

training_order = train_stream.create_epoch_order(seed=1337)

running_loss = 0
initial_training_loss = None
final_training_loss = None

# Puts the model in training mode
model.train()
if device == "cuda":
    torch.cuda.synchronize()
training_start_time = time.perf_counter()

for step in range(num_steps):
    first_block = step * batch_size
    last_block = first_block + batch_size
    block_ids = training_order[first_block:last_block]

    inputs, targets = train_stream.get_batch(
        block_ids,
        device,
    )

    # Clear old gradients from previous training step
    optimizer.zero_grad(set_to_none=True)

    # logits: (B, T, vocab_size)
    # loss: scalar
    logits, loss = model(inputs, targets)

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

    if (step + 1) % 10 == 0:
        average_loss = running_loss / 10
        print(f"step {step + 1}: loss {average_loss:.4f}")
        running_loss = 0.0

if device == "cuda":
    torch.cuda.synchronize()
elapsed_time = time.perf_counter() - training_start_time

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

generated_ids = model.generate(prompt_ids, max_new_tokens=300, temperature=0.8, eos_id=eos_id)

print("\nGenerated sample:")
print(decode(generated_ids[0].tolist()))

stopped_at_eos = generated_ids[0, -1].item() == eos_id
generated_tokens = generated_ids.shape[1] - prompt_ids.shape[1]

print(f"Stopped at EOS: {stopped_at_eos}")
print(f"Generated tokens: {generated_tokens}")


checkpoint_directory = Path("checkpoints")
checkpoint_directory.mkdir(exist_ok=True)

checkpoint_path = checkpoint_directory / f"tiny_gpt_run8_step_{num_steps}.pt"

torch.save(
    {
        "step": num_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "tokenizer_path": str(tokenizer_path),
        "eos_id": eos_id,
        "config": {
            "vocab_size": vocab_size,
            "max_seq_len": max_seq_len,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dropout": dropout,
        },
        "average_training_loss": average_training_loss,
        "average_validation_loss": average_validation_loss,
    },
    checkpoint_path,
)
print(f"Saved checkpoint to {checkpoint_path}")
