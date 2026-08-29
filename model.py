import torch
import torch.nn as nn
from torch.nn import functional as F

# d_model is the token embedding size; head_dim is one head's size.

class AttentionHead(nn.Module):
    def __init__(self, d_model: int, head_dim: int, dropout: float = 0.0):
        super().__init__()

        self.d_model = d_model
        self.head_dim = head_dim
        self.attention_dropout = nn.Dropout(dropout)

        # Matrix of embedding size and head size
        self.q_proj = nn.Linear(d_model, head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, head_dim, bias=False)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        # returns: (B, T, head_dim) or (B, T, D)
        # D is head_dim

        B, T, C = x.shape

        if C != self.d_model:
            raise ValueError("C must be equal to d_model")

        # Each projection maps (B, T, d_model) -> (B, T, head_dim).
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Scaled query-key dot product: (B, T, D) @ (B, D, T) -> (B, T, T)
        out = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Prevent each token from attending to future tokens.
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        # ~ does boolean negation
        out = out.masked_fill(~mask, float('-inf'))
        # Apply softmax across each key of one query
        # Selecting the dimension to be summed up into softmax (Use last dimension when calculating the denominator)
        out = torch.softmax(out, dim=-1)
        out = self.attention_dropout(out)

        # Weighted sum of values: (B, T, T) @ (B, T, D) -> (B, T, D)
        out = out @ v
        return out


class ReferenceMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.heads = nn.ModuleList([
            AttentionHead(d_model, self.head_dim, dropout)
            for _ in range(n_heads)
        ])
        # The linear layer
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each head returns (B, T, head_dim); concatenation restores d_model.
        # -1 is the last dimension with head_dim, concatenate all of them to get d_model (sum)
        # [head 0 features | head 1 features | ... | head 5 features]
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        # out_proj mixes information from different heads
        return self.out_proj(out)

# Vectorized MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.0):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3*d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attention_dropout = nn.Dropout(dropout)

        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask, persistent=False)
    
    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        
        # Split them into three (B, T, C) chunks
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, T, C) -> (B, T, H, D) -> (B, H, T, D)
        # You can split it since H * D = C
        q = q.view(B, T, self.n_heads, self.head_dim)
        q = q.transpose(1,2)

        k = k.view(B, T, self.n_heads, self.head_dim)
        k = k.transpose(1,2)

        v = v.view(B, T, self.n_heads, self.head_dim)
        v = v.transpose(1,2)

        # (B, H, T, D) @ (B, H, D, T) --> (B, H, T, T)
        scores = q @ k.transpose(-2, -1)
        scores = scores / self.head_dim**0.5

        # (T, T) broadcasts across B and H.
        mask = self.causal_mask[:T, :T]
        scores = scores.masked_fill(
            ~mask, float("-inf")
        )

        weights = torch.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)

        # (B, H, T, T) @ (B, H, T, D) --> (B, H, T, D)
        out = weights @ v

        # (B, H, T, D) --> (B, T, H, D)
        out = out.transpose(1,2)

        # transpose() changes the logical order without rearranging
        # memory. contiguous() creates correctly ordered memory so
        # view() can combine H and D.
        out = out.contiguous().view(B, T, C)

        return self.out_proj(out)





class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()

        self.d_model = d_model

        self.net = nn.Sequential(
            nn.Linear(d_model, 4*d_model, bias=False),
            nn.GELU(),
            nn.Linear(4*d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.0):
        super().__init__()

        self.d_model = d_model

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout)
        self.feed_forward = FeedForward(d_model, dropout)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        # max context length
        self.max_seq_len = max_seq_len

        # Embedding for each of the tokens
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Embedding for the sequence order
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, max_seq_len, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Converts each token representation into one score for every possible next token
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Visits every module in TinyGPT and apply the weights
        self.apply(self._init_weights)
        # Share the token embedding and output-classifier weights.
        # For converting each of the vectors to predicted tokens
        # nn.Linear store weights as (out_features, in_features)
        # internally calculates input @ weight.T
        # Linear store weights differently than Embedding
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, token_ids, targets=None):
        B, T = token_ids.shape

        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"
            )
        # works like python range (create a range sequence for the sequence)
        positions = torch.arange(T, device=token_ids.device)
        token_embeddings = self.token_embedding(token_ids)
        position_embeddings = self.position_embedding(positions)

        # (B, T, C) + (T, C); positions broadcast across the batch.
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        # Help keeps value going to output layer consistently scaled
        x = self.final_norm(x)
        # Language model output head
        logits = self.lm_head(x)

        loss = None
        # Use cross entropy for loss function
        # Cross entropy expects predictions: (number_of_examples, v)
        # targets: (number_of_examples)
        if targets is not None:
            loss = F.cross_entropy(
                # can use logits.reshape(-1, self.vocab_size)
                # -1 means let pytorch infer the dimension
                logits.reshape(B * T, self.vocab_size),
                targets.reshape(B * T),
            )

        return logits, loss

    def _init_weights(self, module):
        # Check if the module is linear or embedding
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(
                module.weight,
                mean = 0.0,
                std = 0.02,
            )
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    
    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens, temperature=1.0, eos_id=None):

        if temperature <= 0:
            raise ValueError("temperature must be positive")

        # Generate number of max_new_tokens
        for _ in range(max_new_tokens):
            # keep context that fits in the model
            # index every batch from -max sequence length to the end
            context = token_ids[:, -self.max_seq_len:]

            # This runs self.forward(context) by calling on its own
            # logits returns (B, T, V)
            logits, _ = self(context)

            # Final position predicts the next token
            next_token_logits = logits[:, -1, :]
            # Temperature control randomness (lower temperature make difference larger and vice versa)
            next_token_logits = next_token_logits / temperature

            # Softmax along the vocabulary dimension
            probabilities = F.softmax(next_token_logits, dim=-1)

            # Randomly sample index based on probabilities
            next_token = torch.multinomial(probabilities, num_samples=1)

            # Concatenate the new token to token_ids
            token_ids = torch.cat([token_ids, next_token], dim=1)

            if eos_id is not None and next_token.item() == eos_id:
                break

        return token_ids

