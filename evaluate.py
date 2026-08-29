import torch

from model import TinyGPT

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(
    "checkpoints/tiny_gpt_run5_step_50000.pt",
    map_location = device,
)

stoi = checkpoint["stoi"]
itos = {token_id: token for token, token_id in stoi.items()}
eos_id = stoi["<EOS>"]

model = TinyGPT(**checkpoint["config"]).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# def encode(text):
#     return [stoi[character] for character in text]


# def decode(token_ids):
#     return "".join(
#         itos[token_id]
#         for token_id in token_ids
#         if token_id != eos_id
#     )

# Makes the generation test reproducible.
torch.manual_seed(2026)
torch.cuda.manual_seed_all(2026)

prompt = "Once upon a time"
number_of_samples = 20
max_new_tokens = 500

eos_stops = 0
# Store all stories length
generated_lengths = []

for sample_number in range(number_of_samples):
    prompt_ids = torch.tensor(
        [encode(prompt)],
        dtype=torch.long,
        device=device,
    )

    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        eos_id=eos_id,
    )

    # Remove the prompt so only newly generated IDs remain.
    new_ids = generated_ids[0, prompt_ids.shape[1]:].tolist()

    stopped_at_eos = len(new_ids) > 0 and new_ids[-1] == eos_id

    if stopped_at_eos:
        eos_stops += 1
        text_length = len(new_ids) - 1  # Don't count EOS.
    else:
        text_length = len(new_ids)

    generated_lengths.append(text_length)

    # Print only the first three stories.
    if sample_number < 3:
        print(f"\nSample {sample_number + 1}")
        print(f"Stopped at EOS: {stopped_at_eos}")
        print(decode(generated_ids[0].tolist()))


print("\nEOS evaluation")
print(f"EOS stops: {eos_stops}/{number_of_samples}")
print(f"Stop rate: {eos_stops / number_of_samples:.1%}")
print(f"Average generated length: {sum(generated_lengths) / len(generated_lengths):.1f}")