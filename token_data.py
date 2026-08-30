# Using memory map to treat a disk file like an array without loading it to RAM
import numpy as np
import torch

class TokenStream:
    def __init__(self, path, sequence_length):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.sequence_length = sequence_length
        
        # Calculates how mnay complete training blocks fit in the token stream
        self.number_of_blocks = (len(self.tokens) -1) // sequence_length

    def create_epoch_order(self, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        # Just permutate the number around once
        # might return something like tensor([3, 0, 4, 1, 2]) if randperm(5)
        # Every number appears exactly once but in random order
        return torch.randperm(self.number_of_blocks, generator=generator)
    
    def get_batch(self, block_ids, device):
        chunks = []

        # block_id selects one training block
        for block_id in block_ids.tolist():
            start = block_id * self.sequence_length
            end = start + self.sequence_length + 1

            chunk = np.asarray(self.tokens[start:end], dtype=np.int64)

            chunks.append(chunk)
        
        # Stack all of these blocks into a tensor
        chunks = np.stack(chunks)

        # Shift based on inputs or target
        inputs = torch.from_numpy(chunks[:, :-1])
        targets = torch.from_numpy(chunks[:, 1:])

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        return inputs, targets
