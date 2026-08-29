# Train BPE tokenizer to size 8000
import json
from collections import Counter
from datasets import load_from_disk

def count_pairs(sequences):
    counts = Counter()

    # Sequences is batches of stories
    for token_ids in sequences:
        counts.update(zip(token_ids, token_ids[1:]))
    
    return counts

# Literally merge the pair with each other after looping through token_ids
def merge_pair(token_ids, pair, new_id):
    merged = []
    index = 0

    while index < len(token_ids):
        # Check
        pair_matches = (
            index < len(token_ids) - 1
            and token_ids[index] == pair[0]
            and token_ids[index + 1] == pair[1]
        )

        if pair_matches:
            merged.append(new_id)
            index += 2
        else:
            merged.append(token_ids[index])
            index += 1
    return merged


def train_bpe(stories, vocab_size):
    if vocab_size <= 257:
        raise ValueError("vocab_size must leave room for merges and EOS")
    
    # All currently encoded training stories
    sequences = [list(story.encode("utf-8")) for story in stories]

    merges = {}

    for new_id in range(256, vocab_size - 1):
        pair_counts = count_pairs(sequences)

        if not pair_counts:
            break

        # Which pair_counts has the most count
        most_common_pair = max(pair_counts, key=pair_counts.get)

        # Change the sequences after merging tokens
        sequences = [
            merge_pair(token_ids, most_common_pair, new_id)
            for token_ids in sequences
        ]

        merges[most_common_pair] = new_id

        if new_id % 100 == 0:
            frequency = pair_counts[most_common_pair]
            print(
                f"Created token {new_id}: "
                f"{most_common_pair}, frequency={frequency:,}"
            )

    eos_id = vocab_size - 1

    return merges, eos_id

def build_vocab(merges):
    vocab = {
        token_id: bytes([token_id]) for token_id in range(256)
    }

    sorted_merges = sorted(merges.items(), key=lambda item: item[1])

    for pair, new_id in sorted_merges:
        left_id, right_id = pair
        vocab[new_id] = vocab[left_id] + vocab[right_id]

    return vocab

def encode(text, merges):
    token_ids = list(text.encode("utf-8"))

    while len(token_ids) >= 2:
        # Concatenate every adjacent pair
        current_pairs = set(zip(token_ids, token_ids[1:]))

        # Check if they're in merges or learned yet
        # Smaller number represent merges learned earlier
        pair = min(
            current_pairs, key=lambda candidate: merges.get(candidate, float("inf"))
        )
        
        # Check if they're in merges or learned yet
        if pair not in merges:
            break
        
        # Encode the token_ids with merged pairs
        token_ids = merge_pair(token_ids, pair, merges[pair])
    return token_ids


def decode(token_ids, vocab, eos_id=None):
    token_bytes = b"".join(vocab[token_id] for token_id in token_ids if token_id != eos_id)

    return token_bytes.decode("utf-8", errors="replace")


def save_tokenizer(path, merges, eos_id, vocab_size):
    tokenizer_data = {
        "vocab_size": vocab_size,
        "eos_id": eos_id,
        "merges": [
            [pair[0], pair[1], new_id]
            for pair, new_id in sorted(
                merges.items(),
                key=lambda item: item[1],
            )
        ]
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(tokenizer_data, file, indent=2)

def load_tokenizer(path):
    with open(path, "r", encoding="utf-8") as file:
        tokenizer_data = json.load(file)

    merges = {
        (left_id, right_id): new_id
        for left_id, right_id, new_id
        in tokenizer_data["merges"]
    }

    eos_id = tokenizer_data["eos_id"]
    vocab_size = tokenizer_data["vocab_size"]
    vocab = build_vocab(merges)

    return merges, vocab, eos_id, vocab_size


if __name__ == "__main__":
    train_ds = load_from_disk("data/TinyStories/train")
    training_stories = train_ds["text"][:1000]

    merges, eos_id = train_bpe(training_stories, vocab_size = 300)

    vocab = build_vocab(merges)

    test_text = "Once upon a time, café 😊\n"
    token_ids = encode(test_text, merges)
    decoded_text = decode(token_ids, vocab, eos_id)

    assert decoded_text == test_text

    save_tokenizer(
        "bpe_tokenizer.json",
        merges,
        eos_id,
        vocab_size=300,
    )

    print("Round-trip test passed")