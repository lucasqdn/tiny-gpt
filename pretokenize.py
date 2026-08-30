import json
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

from tokenizer import encode, load_tokenizer

tokenizer_path = Path("tinystories_bpe_2048.json")
output_directory = Path("data/TinyStories/tokenized")
output_directory.mkdir(parents=True, exist_ok=True)

merges, _, eos_id, vocab_size = load_tokenizer(tokenizer_path)

if vocab_size > 65_536:
    raise ValueError("Vocabulary does not fit inside uint16")

def pretokenize_split(dataset_path, output_path):
    dataset = load_from_disk(dataset_path)
    total_tokens = 0

    with open(output_path, "wb") as output_file:
        for row in tqdm(dataset):
            story_ids = encode(row["text"], merges)
            story_ids.append(eos_id)

            token_array = np.asarray(
                story_ids,
                dtype=np.uint16,
            )

            token_array.tofile(output_file)
            total_tokens += len(token_array)
    return total_tokens

train_tokens = pretokenize_split(
    "data/TinyStories/train",
    output_directory / "train.bin",
)

validation_tokens = pretokenize_split(
    "data/TinyStories/validation",
    output_directory / "validation.bin",
)

# For uint16, each token consumes 2 bytes where as int64 would expand it 4 times
metadata = {
    "dtype": "uint16",
    "vocab_size": vocab_size,
    "eos_id": eos_id,
    "tokenizer_path": str(tokenizer_path),
    "train_tokens": train_tokens,
    "validation_tokens": validation_tokens,
}

with open(
    output_directory / "metadata.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metadata, file, indent=2)

print(f"Training tokens: {train_tokens:,}")
print(f"Validation tokens: {validation_tokens:,}")