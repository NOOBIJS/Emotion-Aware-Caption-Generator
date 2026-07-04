import torch
from torch import nn
from torch.nn import functional as F
import math
from typing import Tuple


class GPTEmbedding(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(
            num_embeddings=config["vocab_size"],
            embedding_dim=config["d_model"]
        )
        self.positional_encoding = nn.Parameter(
            data=torch.randn(size=(1, config["context_length"], config["d_model"])),
            requires_grad=True
        )
        self.dropout = nn.Dropout(p=config['emb_dropout'])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.token_embedding(tokens)
        return self.dropout(self.positional_encoding[:, :tokens.shape[1], :] + token_embeddings)


class CausalSelfAttnBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        assert config["d_model"] % config["num_heads"] == 0

        self.d_model = config["d_model"]
        self.head_dim = config["d_model"] // config["num_heads"]
        self.num_heads = config["num_heads"]
        self.softmax_eps = config["softmax_eps"]

        self.projection_layer = nn.Linear(self.d_model, self.d_model * 3)
        self.out_layer = nn.Linear(self.d_model, self.d_model)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.d_model)
        self.attn_dropout = nn.Dropout(p=config['attn_dropout'])

    def _safe_softmax(self, x: torch.Tensor) -> torch.Tensor:
        num = torch.exp(x)
        denom = torch.exp(x).sum(dim=-1, keepdims=True) + self.softmax_eps
        return num / denom

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, CTX_LENGTH = x.shape[0], x.shape[1]
        q, k, v = self.projection_layer(x).split(self.d_model, dim=2)
        q = q.view(B, CTX_LENGTH, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, CTX_LENGTH, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, CTX_LENGTH, self.num_heads, self.head_dim).transpose(1, 2)

        q_k_prod = (q @ k.transpose(2, 3)) + attn_mask.unsqueeze(1)
        wts = self._safe_softmax(q_k_prod / math.sqrt(self.head_dim))
        wts = self.attn_dropout(wts)
        attn_outputs = wts @ v
        y = attn_outputs.transpose(1, 2).contiguous().view(B, CTX_LENGTH, -1)
        return self.layer_norm(x + self.out_layer(y))


class CrossAttnBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        assert config["d_model"] % config["num_heads"] == 0

        self.d_model = config['d_model']
        self.num_heads = config['num_heads']
        self.head_dim = config['d_model'] // config['num_heads']
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.projection_layer = nn.Linear(self.d_model, self.d_model)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.d_model)
        self.attn_dropout = nn.Dropout(p=config['attn_dropout'])

    def forward(self, x: torch.Tensor, image_encoding: torch.Tensor) -> torch.Tensor:
        B, CTX_LENGTH, _ = x.shape

        q = self.q_proj(x).view(B, CTX_LENGTH, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(image_encoding).view(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(image_encoding).view(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        wts = F.softmax((q @ k.transpose(2, 3)) / math.sqrt(self.head_dim), dim=-1)
        wts = self.attn_dropout(wts)
        y = wts @ v
        y = y.transpose(1, 2).contiguous().view(B, CTX_LENGTH, -1)
        return self.layer_norm(x + self.projection_layer(y))


class GPTDecoderBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.csa_block = CausalSelfAttnBlock(config)
        self.cross_attn_block = CrossAttnBlock(config)
        # Import MLPBlock from vit.py
        from .vit import MLPBlock
        self.mlp_block = MLPBlock(config)

    def forward(self, x: torch.Tensor, image_encoding: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        csa_out = self.csa_block(x, attn_mask)
        cross_out = self.cross_attn_block(csa_out, image_encoding)
        mlp_out = self.mlp_block(cross_out)
        return mlp_out


class GPTDecoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.decoder_blocks = nn.ModuleList([GPTDecoderBlock(config) for _ in range(config["num_decoders"])])

    def forward(self, x: torch.Tensor, image_encoding: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        for block in self.decoder_blocks:
            x = block(x, image_encoding, attn_mask)
        return x


class GPT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.device = config["device"]
        self.context_length = config["context_length"]
        self.softmax_eps = config["softmax_eps"]
        self.embedding = GPTEmbedding(config)
        self.decoder = GPTDecoder(config)
        self.cls_head = nn.Linear(config["d_model"], config["vocab_size"])
        self.ignore_index = config["ignore_index"]

    def _create_mask(self, context_length: int, attn_mask: torch.Tensor) -> torch.Tensor:
        mask = torch.triu(
            input=torch.ones(size=(context_length, context_length), requires_grad=False) * float("-inf"),
            diagonal=1
        ).unsqueeze(0).repeat(attn_mask.shape[0], 1, 1)
        mask = mask.to(self.device)
        for i in range(mask.shape[0]):
            mask[i, attn_mask[i].logical_not(), :] = float("-inf")
        return mask

    def forward(self, tokens: torch.Tensor, image_encoding: torch.Tensor, attn_mask: torch.Tensor,
                targets: torch.Tensor = None) -> Tuple[torch.Tensor]:
        embeddings = self.embedding(tokens)
        mask = self._create_mask(tokens.shape[1], attn_mask)
        decoder_out = self.decoder(embeddings, image_encoding, mask)
        logits = self.cls_head(decoder_out)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.reshape(-1),
                                   ignore_index=self.ignore_index)
        return logits, loss