import torch
from torch.functional import _return_inverse_false
import torch.nn as nn


class DINOv2(nn.Module):
    def __init__(self, cfg, embed_dim=256):
        super().__init__()
        self.cfg = cfg

        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

        