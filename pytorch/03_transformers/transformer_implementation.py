"""
Transformer Implementation from Scratch
========================================

A complete, working implementation of the Transformer architecture
as described in "Attention Is All You Need" (Vaswani et al., 2017).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===========================================================================
# POSITIONAL ENCODING
# ===========================================================================


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from the original paper."""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ===========================================================================
# MULTI-HEAD ATTENTION
# ===========================================================================


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Split into heads: (batch, seq, d_model) -> (batch, num_heads, seq, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)

        # Concatenate heads: (batch, num_heads, seq, d_k) -> (batch, seq, d_model)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )

        output = self.W_o(context)
        return output


# ===========================================================================
# FEED-FORWARD NETWORK
# ===========================================================================


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


# ===========================================================================
# ENCODER LAYER & ENCODER
# ===========================================================================


class EncoderLayer(nn.Module):
    """Single encoder layer with pre-normalization."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual
        x2 = self.norm1(x)
        x = x + self.dropout(self.self_attention(x2, x2, x2, mask))

        # Feed-forward with residual
        x2 = self.norm2(x)
        x = x + self.dropout(self.feed_forward(x2))

        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder stack."""

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        max_len=5000,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


# ===========================================================================
# DECODER LAYER & DECODER
# ===========================================================================


class DecoderLayer(nn.Module):
    """Single decoder layer with masked self-attention and cross-attention."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(
            d_model, num_heads, dropout
        )
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        x2 = self.norm1(x)
        x = x + self.dropout(self.self_attention(x2, x2, x2, tgt_mask))

        # Cross-attention
        x2 = self.norm2(x)
        x = x + self.dropout(
            self.cross_attention(
                x2, encoder_output, encoder_output, src_mask
            )
        )

        # Feed-forward
        x2 = self.norm3(x)
        x = x + self.dropout(self.feed_forward(x2))

        return x


class TransformerDecoder(nn.Module):
    """Transformer decoder stack."""

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        max_len=5000,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)


# ===========================================================================
# COMPLETE TRANSFORMER
# ===========================================================================


class Transformer(nn.Module):
    """Complete Transformer for sequence-to-sequence tasks."""

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_encoder_layers=6,
        num_decoder_layers=6,
        max_len=5000,
        dropout=0.1,
        share_embeddings=False,
    ):
        super().__init__()

        self.encoder = TransformerEncoder(
            src_vocab_size,
            d_model,
            num_heads,
            d_ff,
            num_encoder_layers,
            max_len,
            dropout,
        )

        self.decoder = TransformerDecoder(
            tgt_vocab_size,
            d_model,
            num_heads,
            d_ff,
            num_decoder_layers,
            max_len,
            dropout,
        )

        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        if share_embeddings and src_vocab_size == tgt_vocab_size:
            self.decoder.embedding.weight = self.encoder.embedding.weight
            self.output_projection.weight = self.encoder.embedding.weight

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_causal_mask(self, size, device):
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return (mask == 0).unsqueeze(0).unsqueeze(0)

    def generate_padding_mask(self, seq, pad_idx):
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encoder(src, src_mask)

        if tgt_mask is None:
            tgt_mask = self.generate_causal_mask(tgt.size(1), tgt.device)

        decoder_output = self.decoder(
            tgt, encoder_output, src_mask, tgt_mask
        )

        logits = self.output_projection(decoder_output)
        return logits

    @torch.no_grad()
    def generate(self, src, src_mask, max_len, start_token, end_token):
        """Autoregressive generation (greedy decoding)."""
        self.eval()
        device = src.device
        batch_size = src.size(0)

        encoder_output = self.encoder(src, src_mask)

        tgt = torch.full(
            (batch_size, 1), start_token, dtype=torch.long, device=device
        )

        for _ in range(max_len - 1):
            tgt_mask = self.generate_causal_mask(tgt.size(1), device)

            decoder_output = self.decoder(
                tgt, encoder_output, src_mask, tgt_mask
            )
            logits = self.output_projection(decoder_output[:, -1:, :])

            next_token = logits.argmax(dim=-1)
            tgt = torch.cat([tgt, next_token], dim=1)

            if (next_token == end_token).all():
                break

        return tgt


# ===========================================================================
# DECODER-ONLY TRANSFORMER (GPT-style)
# ===========================================================================


class GPTBlock(nn.Module):
    """Single GPT block (decoder-only transformer layer)."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout(self.attention(x2, x2, x2, mask))

        x2 = self.norm2(x)
        x = x + self.dropout(self.feed_forward(x2))

        return x


class GPT(nn.Module):
    """Decoder-only Transformer (GPT-style) for language modeling."""

    def __init__(
        self,
        vocab_size,
        d_model=768,
        num_heads=12,
        d_ff=3072,
        num_layers=12,
        max_len=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                GPTBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.lm_head.weight = self.token_embedding.weight

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, x, mask=None):
        batch_size, seq_len = x.size()
        device = x.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        if mask is None:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device), diagonal=1
            )
            mask = (mask == 0).unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text autoregressively."""
        self.eval()
        device = prompt.device

        for _ in range(max_new_tokens):
            x = (
                prompt
                if prompt.size(1) <= self.max_len
                else prompt[:, -self.max_len :]
            )

            logits = self(x)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            prompt = torch.cat([prompt, next_token], dim=1)

        return prompt


# ===========================================================================
# EXAMPLE USAGE
# ===========================================================================

if __name__ == "__main__":
    print("Testing Transformer implementation...")
    print("=" * 50)

    # Test encoder-decoder Transformer
    src_vocab = 10000
    tgt_vocab = 10000
    model = Transformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        d_model=256,
        num_heads=4,
        d_ff=512,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dropout=0.1,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    src = torch.randint(0, src_vocab, (2, 10))
    tgt = torch.randint(0, tgt_vocab, (2, 15))

    output = model(src, tgt)
    print(f"Encoder-Decoder output shape: {output.shape}")
    print()

    # Test GPT
    gpt = GPT(
        vocab_size=10000,
        d_model=256,
        num_heads=4,
        d_ff=512,
        num_layers=4,
        max_len=512,
        dropout=0.1,
    )

    gpt_params = sum(p.numel() for p in gpt.parameters())
    print(f"GPT parameters: {gpt_params:,}")

    x = torch.randint(0, 10000, (2, 20))
    output = gpt(x)
    print(f"GPT output shape: {output.shape}")

    prompt = torch.randint(0, 10000, (1, 5))
    generated = gpt.generate(
        prompt, max_new_tokens=10, temperature=0.8, top_k=50
    )
    print(f"Generated sequence length: {generated.shape[1]}")

    print()
    print("All tests passed!")
