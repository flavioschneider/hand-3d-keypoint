from typing import List, Tuple 

import torch 
from torch import nn, Tensor 
from .graphnets import GraphNet, GraphUNet
from .encoders.resnet_encoder import ResNetEncoder


class GraphPose(nn.Module):

    def __init__(
        self,
        num_keypoints: int = 21,
        encoder_name: str = 'resnet50',
        decoder_features: List[int] = [128, 16],
        lifter_features: List[Tuple[int, int]] = [(15, 4), (7, 8), (4, 16), (2, 32), (1, 64)],
        lifter_bottleneck_size: int = 20 
    ) -> None:
        super().__init__() 
        self.num_keypoints = num_keypoints
        # Image to features 
        self.encoder = ResNetEncoder(
            name = encoder_name,
            pretrained = True,
            depth = 5
        )
        # Image features to 2d points
        self.decoder = GraphNet(
            in_features = self.encoder.out_channels[-1], 
            out_features = 2, 
            blocks_features = decoder_features, 
            num_nodes = num_keypoints 
        )
        # Lift 2d points to 3d points 
        self.lifter =  GraphUNet(
            in_features = (num_keypoints, 2), # (nodes, features)
            out_features = (num_keypoints, 3),
            blocks_features = lifter_features,
            bottleneck_size = lifter_bottleneck_size 
        )

        
    def forward(self, x: Tensor) -> Tensor:
        """
            x: (B, C, H, W) images 
            Output 0: (B, 21, 2) 
            Output 1: (B, 21, 3)
        """
        x = self.encoder(x)[-1]
        x = x.mean(dim = (2, 3)) # Make the features a vector 
        x = x.unsqueeze(1).repeat(1, self.num_keypoints, 1) # Every node in the graph starts with the same features 
        points_2d = self.decoder(x)
        points_3d = self.lifter(points_2d)
        return points_2d, points_3d 