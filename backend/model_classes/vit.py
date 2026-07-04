import torch
from torch import nn
import timm


class PatchEmbeddings(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.conv_patch_layer = nn.Conv2d(
            in_channels=config['channels'],
            out_channels=config['d_model'],
            kernel_size=config['patch_size'],
            stride=config['patch_size']
        )
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patched_images = self.conv_patch_layer(images)
        flattened_patches = self.flatten(patched_images)
        permuted_patches = flattened_patches.permute(0, 2, 1)
        return permuted_patches


class ViTEmbedding(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.class_token_embedding = nn.Parameter(
            data=torch.randn(size=(1, 1, config['d_model'])),
            requires_grad=True
        )
        self.positional_embedding = nn.Parameter(
            data=torch.randn(size=(1, config['num_patches'] + 1, config["d_model"])),
            requires_grad=True
        )
        self.patch_embeddings_layer = PatchEmbeddings(config)
        self.dropout = nn.Dropout(p=config['emb_dropout'])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patch_embeddings = self.patch_embeddings_layer(images)
        patch_embeddings_with_class_token = torch.cat(
            tensors=(self.class_token_embedding.repeat(patch_embeddings.shape[0], 1, 1), patch_embeddings),
            dim=1
        )
        return self.dropout(patch_embeddings_with_class_token + self.positional_embedding)


class MSABlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.attn_block = nn.MultiheadAttention(
            embed_dim=config["d_model"],
            num_heads=config["num_heads"],
            batch_first=True,
            dropout=config['attn_dropout']
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=config["d_model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attn_block(x, x, x)
        return self.layer_norm(x + attn_output)


class MLPBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        d_model = config["d_model"]
        self.dense_net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(p=config['mlp_dropout']),
            nn.Linear(d_model * 4, d_model)
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x + self.dense_net(x))


class EncoderBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.msa_block = MSABlock(config)
        self.mlp_block = MLPBlock(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_block(self.msa_block(x))


class Encoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config["num_encoders"])])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ViT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embedding_layer = ViTEmbedding(config)
        self.encoder = Encoder(config)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding_layer(images)
        encoded_vectors = self.encoder(embeddings)
        return encoded_vectors[:, 0, :]