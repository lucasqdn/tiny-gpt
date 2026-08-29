# Train BPE tokenizer to size 8000

from collections import Counter
from datasets import load_from_disk

def count_pairs(sequences):
    counts = Counter()

    for token_ids in sequences:
        counts.update(zip(token_ids, token_ids[1:]))
    
    return counts


def merge_pair(token_ids, pair, new_id):
    merged = []
    index = 0

    while index < len(token_ids):
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
    
    sequences = [list(story.encode("utf-8")) for story in stories]

    merges = {}

    for new_id in range(256, vocab_size - 1):
        pair_counts = count_pairs(sequences)

        if not pair_counts:
            break

        most_common_pair = max(pair_counts, key=pair_counts.get)

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


train_ds = load_from_disk("data/TinyStories/train")

training_stories = train_ds["text"][:1000]

merges, eos_id = train_bpe(training_stories, vocab_size = 300)

print(f"Learned {len(merges)} merges")
print(f"EOS ID: {eos_id}")