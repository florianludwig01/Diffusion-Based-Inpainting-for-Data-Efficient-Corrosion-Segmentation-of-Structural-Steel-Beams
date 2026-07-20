import torch.nn as nn
import torch.nn.functional as F

from transformers import Dinov2Model, DINOv3ViTModel

class DINOv2RustSegmenter(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base", freeze_encoder=True):
        super().__init__()

        self.encoder = Dinov2Model.from_pretrained(model_name)
        hidden_dim = self.encoder.config.hidden_size

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        outputs = self.encoder(pixel_values=x)
        tokens = outputs.last_hidden_state

        # remove CLS token
        patch_tokens = tokens[:, 1:, :]

        patch_h = h // 14
        patch_w = w // 14

        features = patch_tokens.transpose(1, 2).reshape(b, -1, patch_h, patch_w)

        logits = self.decoder(features)

        logits = F.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )

        return logits

class DINOv3RustSegmenter(nn.Module):
    def __init__(self, model_name="facebook/dinov3-vitb16-pretrain-lvd1689m", freeze_encoder=True):
        super().__init__()

        self.encoder = DINOv3ViTModel.from_pretrained(model_name)
        hidden_dim = self.encoder.config.hidden_size
        self.patch_size = self.encoder.config.patch_size
        self.num_register_tokens = self.encoder.config.num_register_tokens

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        outputs = self.encoder(pixel_values=x)
        tokens = outputs.last_hidden_state

        # remove CLS token
        first_patch_index = 1 + self.num_register_tokens
        patch_tokens = tokens[:, first_patch_index:, :]

        patch_h = h // self.patch_size
        patch_w = w // self.patch_size

        features = patch_tokens.transpose(1, 2).reshape(b, -1, patch_h, patch_w)

        logits = self.decoder(features)

        logits = F.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )

        return logits