

from typing import List

import torch
from torch import nn, Tensor
from .encoders.resnet_encoder import ResNetEncoder
from .decoders.posef3_decoder import PoseF3Decoder

class RePoseF3(nn.Module): 
    """
        RePose: made up name for ResNet encoder with PoseF3 decoder.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 21,
        heatmap_size: int = 56,
        encoder_name: str = 'resnet50',
        encoder_depth: int = 5,
        encoder_pretrained: bool = True,
        decoder_mid_channels: int = 128,
        decoder_mid_depth: int = 2,
        decoder_low_to_high_upscales: int = 3,
        decoder_dilations: List[int] = [4, 8, 16] 
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

        self.decoder = PoseF3Decoder(
            in_highres_channels = self.encoder.out_channels[-4],
            in_lowres_channels = self.encoder.out_channels[-1],
            low_to_high_upscales = decoder_low_to_high_upscales,
            out_channels = out_channels,
            mid_channels = decoder_mid_channels,
            mid_depth = decoder_mid_depth,
            dilations = decoder_dilations,
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
    
