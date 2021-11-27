from typing import List, Union, Tuple

import torch 
import torch.nn.functional as F
from torch import nn, Tensor

class Conv2dBlock(nn.Module):

    def __init__(
        self, 
        in_channels: int,
        out_channels: int, 
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,  
        padding: Union[int, Tuple[int, int]] = 0, 
        dilation: Union[int, Tuple[int, int]] = 1, 
        groups: int = 1,
        padding_mode: str = 'zeros',
        use_activation: bool = True, 
        use_batchnorm: bool = True,
        use_separable: bool = False 
    ) -> None:
        super().__init__()
        
        block = []
        if not use_separable:
            block += [nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding,
                dilation = dilation,
                groups = groups,
                padding_mode = padding_mode,
                bias = not use_batchnorm
            )] 
        else:
            block +=[
                nn.Conv2d(
                    in_channels = in_channels,
                    out_channels = in_channels,
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = padding,
                    dilation = dilation,
                    groups = in_channels,
                    bias = False 
                ),
                nn.Conv2d(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = 1,
                    bias = not use_batchnorm
                )
            ]
               
        block += [
            nn.ReLU(inplace = True) if use_activation else nn.Identity(), 
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()  
        ]
        
        self.block = nn.Sequential(*block)
        
    
    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

    
class FeatPoseDecoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = 128,
        mid_blocks: int = 4,
        heatmap_size: int = 112,
        soft_argmax_alpha: float = 100.0
    ) -> None:
        super().__init__()
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.mid_blocks = mid_blocks
        self.heatmap_size = heatmap_size
        self.soft_argmax_alpha = soft_argmax_alpha
        
        self.xy_block = nn.Sequential(*[
            Conv2dBlock(
                in_channels = in_channels if i == 0 else mid_channels,
                out_channels = mid_channels,
                kernel_size = 3,
                padding = 1
            )
            for i in range(mid_blocks)
        ])
        
        self.z_block = nn.Sequential(*[
            Conv2dBlock(
                in_channels = in_channels if i == 0 else mid_channels,
                out_channels = mid_channels,
                kernel_size = 3,
                padding = 1
            )
            for i in range(mid_blocks)
        ])
        
        self.x_layer = nn.Conv1d(
            in_channels = mid_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )
        
        self.y_layer = nn.Conv1d(
            in_channels = mid_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )
        
        self.z_layer = nn.Conv1d(
            in_channels = mid_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )
        
        
    def forward(self, features: Tensor) -> Tensor: 
        """
            In shape: (B, I, S, S).
            · B: batch_size.
            · I: in_channels.
            · S: encoder output resolution. 
            Out shape: (B, O, 3).
            · O: out_channels (number of keypoints).
        """
        # Evaluate x, y axis heatmap features 
        xy_block = self.xy_block(features)
        z_block = self.z_block(features)
        
        x = xy_block.mean(dim=(2))
        x = self.x_layer(x)
        x, x_hm = self.soft_argmax_1d(x)
        
        y = xy_block.mean(dim=(3))
        y = self.y_layer(y)
        y, y_hm = self.soft_argmax_1d(y)
        
        # Evaluate z axis (depth) heatmap features         
        z = z_block.mean(dim=(2))
        z = self.z_layer(z) 
        z, z_hm = self.soft_argmax_1d(z)
        
        xyz = torch.cat((x, y, z), dim=2)      
        xyz_hm = torch.cat((x_hm, y_hm, z_hm), dim=2)
        xyz_hm = xyz_hm.view(-1, self.out_channels, 3, self.heatmap_size)

        return xyz, xyz_hm
        
        
    def soft_argmax_1d(self, x):
        # Differentiable argmax function 
        B, C, L = x.shape
        x = F.softmax(x * self.soft_argmax_alpha, dim=2)
        coord = x * torch.arange(start=0, end=self.heatmap_size).to(x.device)
        coord = coord.sum(dim=2, keepdim=True)
        return coord, x
