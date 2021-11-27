from typing import List

import torch
from torch import nn, Tensor
from .encoders.resnet_encoder import ResNetEncoder
from .decoders.litepose_decoder import LitePoseDecoder

class ReLitePose(nn.Module): 
    """
        RePose: made up name for ResNet encoder with PoseF2 decoder.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 21,
        heatmap_size: int = 112,
        encoder_name: str = 'resnet101',
        encoder_depth: int = 5,
        encoder_pretrained: bool = True,
        decoder_mid_channels: int = 128,
        decoder_mid_depth: int = 2
    ) -> None:
        super().__init__() 
        self.out_channels = out_channels
        self.encoder_depth = encoder_depth
        self.heatmap_size = heatmap_size
        
        self.encoder = ResNetEncoder(
            name = encoder_name,
            pretrained = encoder_pretrained,
            depth = encoder_depth
        )

        self.decoder = LitePoseDecoder(
            in_highres_channels = self.encoder.out_channels[-4],
            in_lowres_channels = self.encoder.out_channels[-1],
            out_channels = out_channels,
            mid_channels = decoder_mid_channels,
            mid_depth = decoder_mid_depth,
            heatmap_size = heatmap_size
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
        x, _ = self.decoder(xs[-1], xs[-4])
        return x

    def forward_with_heatmaps(self, x: Tensor) -> Tensor:
        xs = self.encoder(x) 
        x, heatmaps = self.decoder(xs[-1], xs[-4])
        return x, heatmaps