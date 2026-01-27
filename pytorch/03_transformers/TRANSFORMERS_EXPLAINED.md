# The Transformer Architecture: A Complete Guide

The Transformer architecture, introduced in the paper "Attention Is All You Need" (Vaswani et al., 2017), revolutionized deep learning by replacing recurrence with self-attention. This document explains every component in detail.

---

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Input Embeddings](#input-embeddings)
3. [Positional Encoding](#positional-encoding)
4. [Self-Attention Mechanism](#self-attention-mechanism)
5. [Multi-Head Attention](#multi-head-attention)
6. [Feed-Forward Network](#feed-forward-network)
7. [Layer Normalization & Residual Connections](#layer-normalization--residual-connections)
8. [The Complete Encoder](#the-complete-encoder)
9. [The Complete Decoder](#the-complete-decoder)
10. [Putting It All Together](#putting-it-all-together)
11. [Key Insights](#key-insights)

---

## The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRANSFORMER                                 │
│                                                                  │
│  ┌─────────────────────┐      ┌─────────────────────┐          │
│  │      ENCODER        │      │      DECODER        │          │
│  │                     │      │                     │          │
│  │  ┌───────────────┐  │      │  ┌───────────────┐  │          │
│  │  │ Encoder Block │  │      │  │ Decoder Block │  │          │
│  │  │      ×N       │──┼──────┼─▶│      ×N       │  │          │
│  │  └───────────────┘  │      │  └───────────────┘  │          │
│  │         ▲           │      │         ▲           │          │
│  │  ┌──────┴────────┐  │      │  ┌──────┴────────┐  │          │
│  │  │   Pos Embed   │  │      │  │   Pos Embed   │  │          │
│  │  └───────────────┘  │      │  └───────────────┘  │          │
│  │         ▲           │      │         ▲           │          │
│  │  ┌──────┴────────┐  │      │  ┌──────┴────────┐  │          │
│  │  │   Embedding   │  │      │  │   Embedding   │  │          │
│  │  └───────────────┘  │      │  └───────────────┘  │          │
│  │         ▲           │      │         ▲           │          │
│  └─────────┼───────────┘      └─────────┼───────────┘          │
│            │                            │                       │
│      Input Tokens                 Output Tokens                 │
│   (source sequence)            (target sequence)                │
└─────────────────────────────────────────────────────────────────┘
```

**Why Transformers?**

| Problem with RNNs | Transformer Solution |
|-------------------|---------------------|
| Sequential processing (slow) | Parallel processing |
| Long-range dependencies hard to learn | Direct attention to any position |
| Vanishing gradients | Constant path length between positions |
| Hard to parallelize on GPUs | Highly parallelizable |

---

## Input Embeddings

The first step is converting discrete tokens (words, subwords, or characters) into continuous vectors.

```
Token: "The" → Index: 256 → Embedding: [0.21, -0.34, 0.12, ..., 0.45]
                              └──────────────────────────────────────┘
                                          d_model dimensions
                                       (typically 512 or 768)
```

**PyTorch Implementation:**

```python
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        # Scale embeddings by sqrt(d_model) as per original paper
        return self.embedding(x) * math.sqrt(self.d_model)
```

**Why scale by √d_model?**
- Embeddings are initialized with variance ~1/d_model
- Positional encodings have variance ~1
- Scaling brings them to similar magnitudes for addition

---

## Positional Encoding

Since attention has no inherent notion of order, we must inject position information.

### Sinusoidal Positional Encoding

The original paper uses sine and cosine functions of different frequencies:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Visualization:**

```
Position 0: [sin(0), cos(0), sin(0), cos(0), ...]
Position 1: [sin(1/1), cos(1/1), sin(1/10000^(2/512)), cos(...), ...]
Position 2: [sin(2/1), cos(2/1), sin(2/10000^(2/512)), cos(...), ...]
    ...

Each position gets a unique "fingerprint" of sine waves at different frequencies
```

**Why sinusoidal?**

1. **Bounded values**: Always in [-1, 1]
2. **Unique encodings**: Each position has a distinct pattern
3. **Relative positions**: For any fixed offset k, PE(pos+k) can be represented as a linear function of PE(pos)
4. **Extrapolation**: Can handle sequences longer than seen during training

**PyTorch Implementation:**

```python
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # add batch dimension: (1, max_len, d_model)

        self.register_buffer('pe', pe)  # not a parameter, but saved in state_dict

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```

### Learned Positional Embeddings

Modern models (BERT, GPT) often use learned positional embeddings:

```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pos_embedding(positions)
```

---

## Self-Attention Mechanism

Self-attention is the core innovation of the Transformer. It allows each position to attend to all positions in the sequence.

### Intuition

For the sentence "The cat sat on the mat because it was tired":
- When processing "it", attention helps determine that "it" refers to "cat"
- Each word can "look at" every other word to gather context

### The Query-Key-Value Framework

Think of it like a **soft dictionary lookup**:

```
Traditional Dictionary:
    key: "apple" → value: "a red fruit"
    Query "apple" → exact match → returns "a red fruit"

Attention (Soft Lookup):
    Query "fruit" → computes similarity to ALL keys
                 → returns weighted combination of ALL values
```

**The three projections:**

| Component | Analogy | Purpose |
|-----------|---------|---------|
| **Query (Q)** | "What am I looking for?" | Represents current position's request |
| **Key (K)** | "What do I contain?" | Represents what each position offers |
| **Value (V)** | "What information do I provide?" | The actual content to aggregate |

### Mathematical Formulation

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**Step by step:**

```
1. Input X: (seq_len, d_model)

2. Project to Q, K, V:
   Q = X × W_Q    →  (seq_len, d_k)
   K = X × W_K    →  (seq_len, d_k)
   V = X × W_V    →  (seq_len, d_v)

3. Compute attention scores:
   scores = Q × K^T    →  (seq_len, seq_len)

4. Scale (prevents softmax saturation):
   scaled_scores = scores / √d_k

5. Apply softmax (normalize to probabilities):
   attention_weights = softmax(scaled_scores)    →  (seq_len, seq_len)

6. Aggregate values:
   output = attention_weights × V    →  (seq_len, d_v)
```

**Visual representation:**

```
          Input Sequence
    ┌─────────────────────────┐
    │  The   cat   sat   on   │
    └─────────────────────────┘
              │
    ┌─────────┴─────────┐
    ▼         ▼         ▼
┌───────┐ ┌───────┐ ┌───────┐
│   Q   │ │   K   │ │   V   │
└───────┘ └───────┘ └───────┘
    │         │         │
    └────┬────┘         │
         ▼              │
    ┌─────────┐         │
    │  Q×K^T  │  Attention Scores
    └────┬────┘         │
         │              │
         ▼              │
    ┌─────────┐         │
    │ softmax │  Attention Weights
    └────┬────┘         │
         │              │
         └──────┬───────┘
                ▼
         ┌───────────┐
         │ weights×V │  Weighted Sum
         └───────────┘
                │
                ▼
          Output Sequence
```

### Why √d_k Scaling?

The dot product Q·K grows with dimension d_k. For large d_k:
- Dot products become very large
- Softmax pushes values to extremes (0 or 1)
- Gradients become very small

Dividing by √d_k keeps the variance at ~1.

**PyTorch Implementation:**

```python
def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Args:
        query: (batch, seq_len, d_k)
        key: (batch, seq_len, d_k)
        value: (batch, seq_len, d_v)
        mask: optional (batch, seq_len, seq_len)
    """
    d_k = query.size(-1)

    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask (for decoder self-attention or padding)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax over last dimension (keys)
    attention_weights = torch.softmax(scores, dim=-1)

    # Compute output
    output = torch.matmul(attention_weights, value)

    return output, attention_weights
```

---

## Multi-Head Attention

Instead of a single attention, we use multiple "heads" that can attend to different aspects.

### Intuition

Different heads can learn to focus on:
- **Head 1**: Subject-verb relationships
- **Head 2**: Adjective-noun relationships
- **Head 3**: Coreference (pronouns to their referents)
- **Head 4**: Syntactic dependencies
- ...and so on

### Architecture

```
                    Input (batch, seq_len, d_model)
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
   ┌─────────┐           ┌─────────┐           ┌─────────┐
   │ Linear  │           │ Linear  │           │ Linear  │
   │  (W_Q)  │           │  (W_K)  │           │  (W_V)  │
   └────┬────┘           └────┬────┘           └────┬────┘
        │                     │                     │
        ▼                     ▼                     ▼
   ┌─────────────────────────────────────────────────────┐
   │              Split into h heads                      │
   │  (batch, seq, d_model) → (batch, h, seq, d_k)       │
   └─────────────────────────────────────────────────────┘
        │                     │                     │
        └──────────┬──────────┴──────────┬──────────┘
                   ▼                     ▼
        ┌─────────────────────────────────────────┐
        │    Scaled Dot-Product Attention × h     │
        │    (parallel for each head)              │
        └─────────────────────┬───────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │         Concatenate all heads            │
        │  (batch, h, seq, d_k) → (batch, seq, d_model) │
        └─────────────────────┬───────────────────┘
                              │
                              ▼
                        ┌─────────┐
                        │ Linear  │
                        │  (W_O)  │
                        └────┬────┘
                             │
                             ▼
               Output (batch, seq_len, d_model)
```

**Key dimensions:**
- `d_model` = 512 (total model dimension)
- `h` = 8 (number of heads)
- `d_k = d_v = d_model / h` = 64 (dimension per head)

**PyTorch Implementation:**

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections (can be done as one big matrix)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1. Linear projections
        Q = self.W_q(query)  # (batch, seq_len, d_model)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. Split into heads: (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Apply attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)

        # 4. Concatenate heads: (batch, num_heads, seq_len, d_k) → (batch, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 5. Final linear projection
        output = self.W_o(context)

        return output
```

---

## Feed-Forward Network

After attention, each position passes through a simple feed-forward network independently.

```
FFN(x) = ReLU(x × W_1 + b_1) × W_2 + b_2
```

Or with GELU (used in BERT, GPT):
```
FFN(x) = GELU(x × W_1 + b_1) × W_2 + b_2
```

**Architecture:**

```
Input: (batch, seq_len, d_model=512)
           │
           ▼
    ┌─────────────┐
    │   Linear    │  d_model → d_ff (512 → 2048)
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │    ReLU     │  (or GELU)
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   Dropout   │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   Linear    │  d_ff → d_model (2048 → 512)
    └──────┬──────┘
           │
           ▼
Output: (batch, seq_len, d_model=512)
```

**Why expand then contract (512 → 2048 → 512)?**
- The expansion provides more capacity for learning complex transformations
- The "bottleneck" design is computationally efficient
- Acts like two matrix factorizations of a larger matrix

**PyTorch Implementation:**

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
```

---

## Layer Normalization & Residual Connections

These two techniques are crucial for training deep Transformers.

### Residual Connections

```
output = x + sublayer(x)
```

**Benefits:**
- Gradients flow directly through the skip connection
- Easier to learn identity mappings
- Enables very deep networks (100+ layers)

### Layer Normalization

Normalizes across the feature dimension (not batch dimension like BatchNorm):

```
LayerNorm(x) = γ × (x - μ) / (σ + ε) + β

where:
  μ = mean across features
  σ = std across features
  γ, β = learnable parameters
```

**Why LayerNorm instead of BatchNorm?**
- Works with variable sequence lengths
- No dependency on batch statistics (important for inference)
- More stable for sequence data

### Pre-Norm vs Post-Norm

**Post-Norm (Original Paper):**
```
x = x + sublayer(x)
x = LayerNorm(x)
```

**Pre-Norm (Modern Practice):**
```
x = x + sublayer(LayerNorm(x))
```

Pre-norm is more stable for training very deep models.

**PyTorch Implementation:**

```python
class TransformerBlock(nn.Module):
    """A single transformer block with pre-norm."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-norm: normalize before sublayer
        # Self-attention with residual
        attn_output = self.attention(
            self.norm1(x), self.norm1(x), self.norm1(x), mask
        )
        x = x + self.dropout(attn_output)

        # Feed-forward with residual
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x
```

---

## The Complete Encoder

The encoder processes the input sequence bidirectionally (each position can attend to all positions).

```
                        Input Tokens
                             │
                             ▼
                    ┌─────────────────┐
                    │ Token Embedding │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Positional Enc. │
                    └────────┬────────┘
                             │
                             ▼
┌──────────────────────────────────────────────┐
│              Encoder Block ×N                 │
│  ┌────────────────────────────────────────┐  │
│  │        Multi-Head Self-Attention       │  │
│  │     (all positions attend to all)      │  │
│  └────────────────────┬───────────────────┘  │
│           Add & Norm  │                      │
│  ┌────────────────────▼───────────────────┐  │
│  │          Feed-Forward Network          │  │
│  └────────────────────┬───────────────────┘  │
│           Add & Norm  │                      │
└───────────────────────┼──────────────────────┘
                        │
                        ▼
              Encoder Output (to Decoder)
```

**PyTorch Implementation:**

```python
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 max_len=5000, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, x, mask=None):
        # Embedding + positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)
```

---

## The Complete Decoder

The decoder is autoregressive: it generates one token at a time, using previously generated tokens.

```
                     Target Tokens (shifted right)
                             │
                             ▼
                    ┌─────────────────┐
                    │ Token Embedding │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Positional Enc. │
                    └────────┬────────┘
                             │
                             ▼
┌──────────────────────────────────────────────┐
│              Decoder Block ×N                 │
│                                              │
│  ┌────────────────────────────────────────┐  │
│  │      MASKED Multi-Head Self-Attention  │  │
│  │   (can only attend to earlier tokens)  │◀─┼─── Causal Mask
│  └────────────────────┬───────────────────┘  │
│           Add & Norm  │                      │
│                       ▼                      │
│  ┌────────────────────────────────────────┐  │
│  │      Multi-Head Cross-Attention        │  │
│  │   Q from decoder, K,V from encoder     │◀─┼─── Encoder Output
│  └────────────────────┬───────────────────┘  │
│           Add & Norm  │                      │
│                       ▼                      │
│  ┌────────────────────────────────────────┐  │
│  │          Feed-Forward Network          │  │
│  └────────────────────┬───────────────────┘  │
│           Add & Norm  │                      │
└───────────────────────┼──────────────────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │    Linear    │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │   Softmax    │
                 └──────────────┘
                        │
                        ▼
              Output Probabilities
```

### The Causal Mask

The decoder must not "cheat" by looking at future tokens during training:

```
Position:      0    1    2    3    4
             ┌────┬────┬────┬────┬────┐
Position 0   │ 1  │ 0  │ 0  │ 0  │ 0  │  Can only see position 0
Position 1   │ 1  │ 1  │ 0  │ 0  │ 0  │  Can see positions 0-1
Position 2   │ 1  │ 1  │ 1  │ 0  │ 0  │  Can see positions 0-2
Position 3   │ 1  │ 1  │ 1  │ 1  │ 0  │  Can see positions 0-3
Position 4   │ 1  │ 1  │ 1  │ 1  │ 1  │  Can see all positions
             └────┴────┴────┴────┴────┘

(1 = can attend, 0 = cannot attend → masked to -∞ before softmax)
```

**PyTorch Implementation:**

```python
def generate_causal_mask(size):
    """Generate causal mask for decoder self-attention."""
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0  # True where attention is allowed


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # Masked self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        # Cross-attention (to encoder output)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        x2 = self.norm1(x)
        x = x + self.dropout(self.self_attention(x2, x2, x2, tgt_mask))

        # Cross-attention: Q from decoder, K,V from encoder
        x2 = self.norm2(x)
        x = x + self.dropout(self.cross_attention(x2, encoder_output, encoder_output, src_mask))

        # Feed-forward
        x2 = self.norm3(x)
        x = x + self.dropout(self.feed_forward(x2))

        return x
```

---

## Putting It All Together

### Complete Transformer Model

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
                 num_heads=8, d_ff=2048, num_encoder_layers=6,
                 num_decoder_layers=6, max_len=5000, dropout=0.1):
        super().__init__()

        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, num_heads, d_ff,
            num_encoder_layers, max_len, dropout
        )

        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, num_heads, d_ff,
            num_decoder_layers, max_len, dropout
        )

        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode source sequence
        encoder_output = self.encoder(src, src_mask)

        # Decode target sequence
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)

        # Project to vocabulary
        logits = self.output_projection(decoder_output)

        return logits
```

### Using PyTorch's Built-in Transformer

```python
# PyTorch provides nn.Transformer for convenience
transformer = nn.Transformer(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    batch_first=True
)

# Forward pass
src = torch.randn(32, 10, 512)  # (batch, src_seq_len, d_model)
tgt = torch.randn(32, 20, 512)  # (batch, tgt_seq_len, d_model)
output = transformer(src, tgt)
```

---

## Key Insights

### 1. Attention Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Self-Attention | O(n² × d) | O(n²) |
| Feed-Forward | O(n × d²) | O(d) |

For long sequences, attention becomes the bottleneck → leads to efficient attention variants (Linformer, Performer, etc.)

### 2. Why Transformers Work So Well

1. **Parallel processing**: All positions computed simultaneously
2. **Long-range dependencies**: Any two positions are one step apart
3. **Flexibility**: Same architecture works for text, images, audio, etc.
4. **Scale**: Benefits significantly from more data and parameters

### 3. Modern Variants

| Model | Architecture | Key Innovation |
|-------|-------------|----------------|
| **BERT** | Encoder-only | Bidirectional, masked language modeling |
| **GPT** | Decoder-only | Autoregressive, causal attention |
| **T5** | Encoder-Decoder | Text-to-text framework |
| **ViT** | Encoder-only | Treats images as sequences of patches |

### 4. Training Tips

- **Learning rate warmup**: Gradually increase LR at start of training
- **Label smoothing**: Softens one-hot labels for better generalization
- **Dropout**: Apply to attention weights, embeddings, and FF layers
- **Weight tying**: Share embedding and output projection weights

---

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need"
2. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers"
3. Radford, A., et al. (2018). "Improving Language Understanding by Generative Pre-Training" (GPT)
4. The Annotated Transformer: https://nlp.seas.harvard.edu/2018/04/03/attention.html
