from typing import List

import torch
from torch import nn, Tensor
from .encoders.effnet_encoder import EfficientNetEncoder
from .decoders.posef1_decoder import PoseF1Decoder

class EffPose(nn.Module): 
    """
        EffPose: made up name for EfficientNetV2 encoder with Pose decoder.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 21,
        heatmap_size: int = 112,
        encoder_name: str = 'efficientnet-b6',
        decoder_mid_channels: int = 256,
        decoder_mid_blocks: int = 4,
        decoder_soft_argmax_alpha: float = 1.0
    ) -> None:
        super().__init__() 
        self.out_channels = out_channels
        
        self.encoder = EfficientNetEncoder(
            name = encoder_name
        )

        self.decoder = PoseF1Decoder(
            in_channels = self.encoder.out_channels[-1],
            out_channels = out_channels,
            mid_channels = decoder_mid_channels,
            mid_blocks = decoder_mid_blocks,
            heatmap_size = heatmap_size,
            soft_argmax_alpha = decoder_soft_argmax_alpha,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
            In shape: (B, I, H, W).
            · B: batch_size.
            · I: in_channels (usually 3 for RGB)
            · H: image height.
            · W: image width.
            Out shape: (B, O, 3).
            · O: out_channels (number of keypoints).
        """
        xs = self.encoder(x) 
        x, _ = self.decoder(xs[-1])
        return x

    def forward_with_heatmaps(self, x: Tensor) -> Tensor:
        xs = self.encoder(x) 
        x, heatmaps = self.decoder(xs[-1])
        return x, heatmaps