from datasets import load_dataset

ds = load_dataset("roneneldan/TinyStories")
ds.save_to_disk("data/TinyStories")
print("Dataset saved to data/TinyStories")