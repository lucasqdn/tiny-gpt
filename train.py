import torch
import random
import time
from datasets import load_from_disk
from model import TinyGPT
from values import d_model, dropout, max_seq_len, n_heads, n_layers

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

train_ds = load_from_disk("data/TinyStories/train")
validation_ds = load_from_disk("data/TinyStories/validation")
print("Training and validation datasets loaded successfully.")

random.seed(1337)
torch.manual_seed(1337)
torch.cuda.manual_seed_all(1337)

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

batch_size = 8
sequence_length = 128

# Create one group of training examples
# 1 input pair with 1 target
def get_batch(dataset):
    # inputs: (batch_size, sequence_length)
    # target: (batch_size, sequence_length)

    inputs = []
    targets = []

    # Keep sampling until enough batches
    while len(inputs) < batch_size:
        story_index = random.randrange(len(dataset))
        story = dataset[story_index]["text"]
        token_ids = encode(story)

        # Ignore stories that are too short
        if len(token_ids) < sequence_length + 1:
            continue

        # Choose random starting position
        # len(token_ids) - seq_length - 1 is the last position you can pick before it overflows
        start = random.randint(0, len(token_ids) - sequence_length - 1)

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


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)

num_steps = 1_000
running_loss = 0
initial_training_loss = None
final_training_loss = None

# Puts the model in training mode
model.train()

if device == "cuda":
    torch.cuda.synchronize()
training_start_time = time.perf_counter()

for step in range(num_steps):
    inputs, targets = get_batch(train_ds)

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
model.eval()
validation_losses = []

with torch.no_grad():
    for _ in range(20):
        validation_inputs, validation_targets = get_batch(validation_ds)
        _, validation_loss = model(validation_inputs, validation_targets)
        validation_losses.append(validation_loss.item())

average_validation_loss = sum(validation_losses) / len(validation_losses)

print("\nRun summary")
print(f"Initial training loss: {initial_training_loss:.4f}")
print(f"Final training loss: {final_training_loss:.4f}")
print(f"Average validation loss: {average_validation_loss:.4f}")
print(f"Elapsed training time: {elapsed_time:.2f} seconds")
print(f"Tokens per second: {tokens_per_second:,.0f}")
print(f"Total tokens trained: {total_tokens_trained:,}")
