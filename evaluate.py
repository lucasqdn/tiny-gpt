import torch

from model import TinyGPT
from tokenizer import decode, encode, load_tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(
    "checkpoints/tiny_gpt_run6_step_50000.pt",
    map_location=device,
)

tokenizer_path = checkpoint["tokenizer_path"]
merges, vocab, eos_id, vocab_size = load_tokenizer(
    tokenizer_path
)

if vocab_size != checkpoint["config"]["vocab_size"]:
    raise ValueError("Tokenizer and model vocabulary sizes do not match")

if eos_id != checkpoint["eos_id"]:
    raise ValueError("Tokenizer and checkpoint EOS IDs do not match")

model = TinyGPT(**checkpoint["config"]).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Makes the generation test reproducible.
torch.manual_seed(2026)
torch.cuda.manual_seed_all(2026)

prompt = "Once upon a time"
number_of_samples = 20
max_new_tokens = 500

eos_stops = 0
generated_token_lengths = []
generated_character_lengths = []

for sample_number in range(number_of_samples):
    prompt_ids = torch.tensor(
        [encode(prompt, merges)],
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

    # EOS is not visible text, so don't count it as a generated token.
    generated_token_length = len(new_ids) - int(stopped_at_eos)
    generated_text = decode(new_ids, vocab, eos_id)

    generated_token_lengths.append(generated_token_length)
    generated_character_lengths.append(len(generated_text))

    # Print only the first three stories.
    if sample_number < 3:
        print(f"\nSample {sample_number + 1}")
        print(f"Stopped at EOS: {stopped_at_eos}")
        print(f"Generated BPE tokens: {generated_token_length}")
        print(decode(generated_ids[0].tolist(), vocab, eos_id))


print("\nEOS evaluation")
print(f"EOS stops: {eos_stops}/{number_of_samples}")
print(f"Stop rate: {eos_stops / number_of_samples:.1%}")
print(
    "Average generated BPE tokens: "
    f"{sum(generated_token_lengths) / len(generated_token_lengths):.1f}"
)
print(
    "Average generated characters: "
    f"{sum(generated_character_lengths) / len(generated_character_lengths):.1f}"
)
