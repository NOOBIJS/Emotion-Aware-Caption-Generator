import torch
from torch import nn
import timm
from typing import Tuple, List
from .vit import ViT
from .gpt import GPT


class ImageCaptionModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.device = config['device']
        self.is_vit_pretrained = False
        if config['vit_kwargs']["pretrained_model_name"] is not None:
            self.is_vit_pretrained = True
            self.vit = timm.create_model(
                model_name=config['vit_kwargs']["pretrained_model_name"],
                pretrained=True,
                num_classes=0,
                global_pool='avg'
            )
            config["vit_kwargs"]["d_model"] = self.vit.embed_dim
        else:
            self.vit = ViT(config['vit_kwargs'])
        self.gpt = GPT(config['gpt_kwargs'])
        self.dimension_mapping_layer = nn.Linear(config["vit_kwargs"]['d_model'], config["gpt_kwargs"]['d_model'])

    def forward(self, image: torch.Tensor, tokens: torch.Tensor, attn_mask: torch.Tensor,
                targets: torch.Tensor = None) -> Tuple[torch.Tensor]:
        image_encoding = self.vit(image)
        dimension_mapped_image_encoding = self.dimension_mapping_layer(image_encoding[:, None, :])
        return self.gpt(tokens, dimension_mapped_image_encoding, attn_mask, targets)

    @torch.inference_mode()
    def generate(self, image: torch.Tensor, sos_token: int, eos_token: int, max_len: int = 40) -> List[int]:
        image_encoding: torch.Tensor = self.vit(image)
        dimension_mapped_image_encoding = self.dimension_mapping_layer(image_encoding[:, None, :])

        tokens = torch.tensor(data=[[sos_token]], requires_grad=False).to(self.device)
        attn_mask = torch.tensor(data=[[1]], requires_grad=False).to(self.device)

        while tokens.shape[1] < max_len and tokens[0, -1] != eos_token:
            logits, _ = self.gpt(tokens, dimension_mapped_image_encoding, attn_mask, None)
            next_token = torch.argmax(logits[0, -1, :], dim=0).item()

            next_token_tensor = torch.tensor([[next_token]], requires_grad=False, device=self.device)
            tokens = torch.cat((tokens, next_token_tensor), dim=-1)

            next_mask_tensor = torch.tensor([[1]], requires_grad=False, device=self.device)
            attn_mask = torch.cat((attn_mask, next_mask_tensor), dim=-1)

        return list(tokens[0])