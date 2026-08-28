import torch
import random
import time
from datasets import load_from_disk
from model import TinyGPT
from values import d_model, dropout, max_seq_len, n_heads, n_layers
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

train_ds = load_from_disk("data/TinyStories/train")
validation_ds = load_from_disk("data/TinyStories/validation")
print("Training and validation datasets loaded successfully.")

random.seed(1337)
torch.manual_seed(1337)
torch.cuda.manual_seed_all(1337)
training_rng = random.Random(1337)
training_eval_rng = random.Random(2024)
validation_rng = random.Random(2025)

# Build one character vocabulary across the entire training split.
characters = set()
for story in train_ds["text"]:
    # update breaks down the words into characters and add them to the set
    characters.update(story)

characters = sorted(characters)
stoi = {character: i for i, character in enumerate(characters)}
itos = {i: character for character, i in stoi.items()}
vocab_size = len(characters)

model = TinyGPT(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    dropout=dropout,
).to(device)

def encode(text):
    return [stoi[character] for character in text]

def decode(token_ids):
    return "".join(itos[token_id] for token_id in token_ids)

batch_size = 8
sequence_length = 128

# Create one group of training examples
# 1 input pair with 1 target
def get_batch(dataset, rng):
    # inputs: (batch_size, sequence_length)
    # target: (batch_size, sequence_length)
    inputs = []
    targets = []

    # Keep sampling until enough batches
    while len(inputs) < batch_size:
        story_index = rng.randrange(len(dataset))
        story = dataset[story_index]["text"]
        token_ids = encode(story)

        # Ignore stories that are too short
        if len(token_ids) < sequence_length + 1:
            continue

        # Choose random starting position
        # len(token_ids) - seq_length - 1 is the last position you can pick before it overflows
        start = rng.randint(0, len(token_ids) - sequence_length - 1)

        # Python slicing excludes the ending index
        chunk = token_ids[start : start + sequence_length + 1]

        # Create shifted inputs and targets
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        inputs.append(x)
        targets.append(y)

    inputs = torch.stack(inputs).to(device)
    targets = torch.stack(targets).to(device)

    return inputs, targets

num_evaluation_batches = 20

fixed_training_batches = [
    get_batch(train_ds, training_eval_rng) for _ in range(num_evaluation_batches)
]

fixed_validation_batches = [
    get_batch(validation_ds, validation_rng) for _ in range(num_evaluation_batches)
]

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)

num_steps = 50_000
running_loss = 0
initial_training_loss = None
final_training_loss = None

# Puts the model in training mode
model.train()

if device == "cuda":
    torch.cuda.synchronize()
training_start_time = time.perf_counter()

for step in range(num_steps):
    inputs, targets = get_batch(train_ds, training_rng)

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

prompt_ids = torch.tensor([[stoi[character] for character in prompt]], dtype=torch.long, device=device)

generated_ids = model.generate(prompt_ids, max_new_tokens=300, temperature=0.8)

print("\nGenerated sample:")
print(decode(generated_ids[0].tolist()))


checkpoint_directory = Path("checkpoints")
checkpoint_directory.mkdir(exist_ok=True)

checkpoint_path = (checkpoint_directory / f"tiny_gpt_step_{num_steps}.pt")

torch.save(
    {
        "step": num_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "stoi": stoi,
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

