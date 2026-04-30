import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.attention_network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim//2, 1)
        )

    def forward(self, x):
        raw_scores = self.attention_network(x)

        weights = F.softmax(raw_scores, dim=1)

        weighted_frames = x * weights

        pooled_representation = torch.sum(weighted_frames, dim=1)

        return pooled_representation



class DINOv2FT(nn.Module):
    def __init__(self, cfg, embed_dim=256, freeze_backbone=True):
        # Set up internal Pytorch mechanisms
        super().__init__()
        self.cfg = cfg

        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

        # Freeze the bacbone weights
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Temporal attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=384, # DinoV2 output size
            nhead=8, # 8 separate attention heads 
            dim_feedforward=1024, # size of hidden FC layer
            batch_first=True, # Data will be shaped [batch, time, features]
            dropout=0.1 # randomly breaks connections to prevent overfittingl
        )

        # Stacks two temporal attention layers
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # positions, learnable tensor
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, self.cfg.num_frames, 384))

        # Metric projection head
        self.projector = nn.Sequential(
            nn.Linear(384, 512),
            nn.GELU(), # Standard for ViTs 
            nn.Linear(512, embed_dim)
        )

        # Attention pooling
        self.attn_pool = AttentionPooling(embed_dim=384)

    def forward(self, x):
        # Batch, time, channels, height, width
        B, T, C, H, W = x.shape

        # Fold time into the batch dimension
        x_flat = x.view(B * T, C, H, W)

        # send through backbone
        frame_features = self.backbone(x_flat)

        # Time unfolding- back to seperate video sequences
        temporal_seq = frame_features.view(B, T, -1)

        # adding the positional embedding 
        temporal_seq = temporal_seq + self.temporal_pos_embed

        # Passes the model through the attention block
        attended_seq = self.temporal_transformer(temporal_seq)

        # Average across the time dimensions
        spatio_temporal_feature = self.attn_pool(attended_seq)

        # normalize the embeddings
        embeddings = self.projector(spatio_temporal_feature)
        normalized_embeddings = F.normalize(embeddings)

        return normalized_embeddings