import torch.nn as nn
import torch.nn.functional as F

from transformers import Dinov2Model

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

        # CLS token entfernen
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