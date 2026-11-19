import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(
            position * div_term[:pe[:, 1::2].shape[1]]
        )
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiheadAttentionLayer(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.attention = nn.MultiheadAttention(
                        embed_dim=d_model,
                        num_heads=num_heads,
                        batch_first=True,
                    )

    def forward(self, x, mask=None):
                    output, weights = self.attention(
                        x, x, x, attn_mask=mask, need_weights=False
                    )
                    return output, weights


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
     super().__init__()
     self.attention = MultiheadAttentionLayer(num_heads, d_model)
     self.norm1 = nn.LayerNorm(d_model)
     self.norm2 = nn.LayerNorm(d_model)
     self.ffn = nn.Sequential(
                        nn.Linear(d_model, d_ff),
                        nn.GELU(),
                        nn.Linear(d_ff, d_model),
                    )

    def forward(self, x, mask=None):
        attended, weights = self.attention(x, mask=mask)
        x = self.norm1(x + attended)
        x = self.norm2(x + self.ffn(x))
        return x, weights


class TinyGPT(nn.Module):
    def __init__(
                    self,
                    vocab_size,
                    d_model=128,
                    num_heads=4,
                    num_layers=4,
                    d_ff=512,
                    max_len=256,
                ):
                    super().__init__()

                    self.embedding = nn.Embedding(vocab_size, d_model)
                    self.pos_encoding = PositionalEncoding(d_model, max_len)
                    self.blocks = nn.ModuleList(
                        [
                            TransformerBlock(d_model, num_heads, d_ff)
                            for _ in range(num_layers)
                        ]
                    )
                    self.ln_final = nn.LayerNorm(d_model)
                    self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
                    seq_len = token_ids.size(1)
                    x = self.pos_encoding(self.embedding(token_ids))

                    # True entries are masked by nn.MultiheadAttention.
                    causal_mask = torch.triu(
                        torch.ones(seq_len, seq_len, device=token_ids.device, dtype=torch.bool),
                        diagonal=1,
                    )

                    for block in self.blocks:
                        x, _ = block(x, mask=causal_mask)

                    return self.lm_head(self.ln_final(x))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

model = TinyGPT(vocab_size=1000)
tokens = torch.randint(0, 1000, (2, 10))
logits = model(tokens)

print(logits.shape)  # torch.Size([2, 10, 1000])
def __init__(self, d_model, max_len=256):
 super().__init__()
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiheadAttentionLayer(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x, mask=None):
        output, _ = self.attention(
            x, x, x, attn_mask=mask, need_weights=False
        )
        return output


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiheadAttentionLayer(num_heads, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, mask=None):
        x = self.norm1(x + self.attention(x, mask=mask))
        x = self.norm2(x + self.ffn(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=4,
        d_ff=512,
        max_len=256,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        seq_len = token_ids.size(1)
        x = self.pos_encoding(self.embedding(token_ids))

        # True entries mask future tokens.
        causal_mask = torch.triu(
            torch.ones(
                seq_len, seq_len,
                device=token_ids.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        for block in self.blocks:
            x = block(x, mask=causal_mask)

        return self.lm_head(self.ln_final(x))


model = TinyGPT(vocab_size=1000)
tokens = torch.randint(0, 1000, (2, 10))
logits = model(tokens)
print(logits.shape)  # torch.Size([2, 10, 1000])
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, mask=None):
        attended, _ = self.attention(
            x, x, x, attn_mask=mask, need_weights=False
        )
        x = self.norm1(x + attended)
        return self.norm2(x + self.ffn(x))


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=4,
        d_ff=512,
        max_len=256,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        seq_len = token_ids.size(1)
        x = self.pos_encoding(self.embedding(token_ids))

        # True entries mask future tokens.
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=token_ids.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        for block in self.blocks:
            x = block(x, mask=causal_mask)

        return self.lm_head(self.ln_final(x))


model = TinyGPT(vocab_size=1000)
tokens = torch.randint(0, 1000, (2, 10))
logits = model(tokens)
print(logits.shape)  # torch.Size([2, 10, 1000])
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(
            position * div_term[:pe[:, 1::2].shape[1]]
        )
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(
            position * div_term[:pe[:, 1::2].shape[1]]
        )
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, mask=None):
        attended, _ = self.attention(
            x, x, x, attn_mask=mask, need_weights=False
        )
        x = self.norm1(x + attended)
        return self.norm2(x + self.ffn(x))


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=4,
        d_ff=512,
        max_len=256,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        seq_len = token_ids.size(1)
        x = self.pos_encoding(self.embedding(token_ids))

        # True entries mask future tokens.
        causal_mask = torch.triu(
            torch.ones(
                seq_len, seq_len,
                device=token_ids.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        for block in self.blocks:
            x = block(x, mask=causal_mask)

        return self.lm_head(self.ln_final(x))


model = TinyGPT(vocab_size=1000)
tokens = torch.randint(0, 1000, (2, 10))
logits = model(tokens)
print(logits.shape)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(
            position * div_term[:pe[:, 1::2].shape[1]]
        )
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
                def __init__(self, d_model, num_heads, d_ff):
                    super().__init__()
                    self.attention = MultiheadAttentionlayer(num_heads, d_model)
                    self.norm1 = nn.LayerNorm(d_model)
                    self.norm2 = nn.LayerNorm(d_model)
                    self.ffn = nn.Sequential(
                        nn.Linear(d_model, d_ff),
                        nn.GELU(),
                        nn.Linear(d_ff, d_model),
                    )

                def forward(self, x, mask=None):
                    attended, weights = self.attention(x, mask=mask)
                    x = self.norm1(x + attended)
                    x = self.norm2(x + self.ffn(x))
                    return x, weights


class TinyGPT(nn.Module):
                def __init__(
                    self,
                    vocab_size,
                    d_model=128,
                    num_heads=4,
                    num_layers=4,
                    d_ff=512,
                    max_len=256,
                ):
                    super().__init__()

                    self.embedding = nn.Embedding(vocab_size, d_model)
                    self.pos_encoding = PositionalEncoding(d_model, max_len)
                    self.blocks = nn.ModuleList(
                        [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
                    )
                    self.ln_final = nn.LayerNorm(d_model)
                    self.lm_head = nn.Linear(d_model, vocab_size)

                def forward(self, token_ids):
                    _, seq_len = token_ids.shape
                    x = self.pos_encoding(self.embedding(token_ids))

                    causal_mask = torch.tril(
                        torch.ones(seq_len, seq_len, device=token_ids.device)
                    )[None, None, :, :]

                    for block in self.blocks:
                        x, _ = block(x, mask=causal_mask)

                    return self.lm_head(self.ln_final(x))

# Fix the typo in the class name used by TransformerBlock
# (MultiheadAttentionLayer vs MultiheadAttentionlayer)
# Fix the typo in the class name used by TransformerBlock
# (MultiheadAttentionLayer vs MultiheadAttentionlayer)
class MultiheadAttentionLayer(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x, mask=None):
        output, weights = self.attention(
            x, x, x,
            attn_mask=mask,
            need_weights=False,
        )
        return output, weights

# Recreate the missing model pieces with the correct class names.
# This overwrites the broken/duplicated definitions from earlier cells.
class MultiheadAttentionLayer(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x, mask=None):
        attn_mask = mask
        if mask is not None and mask.dim() == 2:
            attn_mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,T,T]
        output, _ = self.attention(
            x, x, x, attn_mask=attn_mask, need_weights=False
        )
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiheadAttentionLayer(num_heads, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, mask=None):
        attended = self.attention(x, mask=mask)
        x = self.norm1(x + attended)
        return self.norm2(x + self.ffn(x))


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=4,
        d_ff=512,
        max_len=256,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        x = self.pos_encoding(self.embedding(token_ids))
        seq_len = token_ids.size(1)

        # True entries mask future tokens.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=token_ids.device, dtype=torch.bool),
            diagonal=1,
        )

        for block in self.blocks:
            x = block(x, mask=causal_mask)

        return self.lm_head(self.ln_final(x))


model = TinyGPT(vocab_size=1000)
tokens = torch.randint(0, 1000, (2, 10))
def _fixed_attention_forward(self, x, mask=None):
    output, _ = self.attention(
        x, x, x, attn_mask=mask, need_weights=False
    )
    return output

for block in model.blocks:
    block.attention.forward = MethodType(_fixed_attention_forward, block.attention)

logits = model(tokens)
print(logits.shape)  # torch.Size([2, 10, 1000])
tokens = torch.randint(0, 1000, (2, 10))
logits = model(tokens)
print(logits.shape)  # torch.Size([2, 10, 1000])
tokens = torch.randint(0, 1000, (2, 10))
logits = model(tokens)

print(logits.shape)  # torch.Size([2, 10, 1000])
tokens = torch.randint(0, 1000, (2, 10))
logits = model(tokens)

print(logits.shape)  # torch.Size([2, 10, 1000])
def forward(self, x):
    return x + self.pe[:, : x.size(1)]

class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=4,
        d_ff=512,
        max_len=256,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        x = self.pos_encoding(self.embedding(token_ids))
        seq_len = token_ids.size(1)

        # True entries mask future tokens in nn.MultiheadAttention.
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=token_ids.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        for block in self.blocks:
            x = block(x, mask=causal_mask)

        return self.lm_head(self.ln_final(x))


model = TinyGPT(vocab_size=1000)
tokens = torch.randint(0, 1000, (2, 10))
x = model.pos_encoding(model.embedding(tokens))
seq_len = tokens.size(1)
causal_mask = torch.triu(
    torch.ones(seq_len, seq_len, device=tokens.device, dtype=torch.bool),
    diagonal=1,
)

for block in model.blocks:
    attended, _ = block.attention.attention(
        x, x, x, attn_mask=causal_mask, need_weights=False
    )
    x = block.norm1(x + attended)
    x = block.norm2(x + block.ffn(x))

logits = model.lm_head(model.ln_final(x))
print(logits.shape)  # torch.Size([2, 10, 1000])

import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, mask=None):
        attended, _ = self.attention(
            x, x, x, attn_mask=mask, need_weights=False
        )
        x = self.norm1(x + attended)
        return self.norm2(x + self.ffn(x))


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=4,
        d_ff=512,
        max_len=256,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        x = self.pos_encoding(self.embedding(token_ids))
        seq_len = token_ids.size(1)

        # True entries mask future tokens.
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=token_ids.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        for block in self.blocks:
            x = block(x, mask=causal_mask)

        return self.lm_head(self.ln_final(x))


model = TinyGPT(vocab_size=1000)
tokens = torch.randint(0, 1000, (2, 10))
logits = model(tokens)
print(logits.shape)  # torch.Size([2, 10, 1000])
class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=4,
        d_ff=512,
        max_len=256,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff)
                for _ in range(self.num_layers)
            ]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        x = self.pos_encoding(self.embedding(token_ids))
        seq_len = token_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(
                seq_len, seq_len,
                device=token_ids.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        for block in self.blocks:
            x = block(x, mask=causal_mask)

        return self.lm_head(self.ln_final(x))
self.embedding = nn.Embedding(vocab_size, d_model)
self.pos_encoding = PositionalEncoding(d_model, max_len)
self.blocks = nn.ModuleList(
    [
        TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
        for _ in range(self.num_layers)
    ]
)
self.ln_final = nn.LayerNorm(d_model)
self.lm_head = nn.Linear(d_model, vocab_size)
self.blocks = nn.ModuleList(
    [
        TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
        for _ in range(self.num_layers)
    ]
)
self.ln_final = nn.LayerNorm(d_model)
self.lm_head = nn.Linear(d_model, vocab_size)
self.lm_head = nn.Linear(d_model, vocab_size)

def forward(self, token_ids):
        batch, seq_len = token_ids.shape
   
        x = self.embedding(token_ids)
        x = self.pos_encoding(x)

        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=token_ids.device)
        ).unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x, _ = block(x, mask=causal_mask)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits

model = TinyGPT(vocab_size=1000)

tokens = torch.randint(0, 1000, (2, 10))
logits = model(tokens)

print(logits.shape)  # torch.Size([2, 10, 1000])