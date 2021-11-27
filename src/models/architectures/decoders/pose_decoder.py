from typing import List, Union, Tuple

import torch 
import torch.nn.functional as F
from torch import nn, Tensor
from src.utils.data_utils import Scale
    

class Conv1dBlock(nn.Module):

    def __init__(
        self, 
        in_channels: int,
        out_channels: int, 
        kernel_size: int, 
        padding: int = 0, 
        stride: int = 1, 
        dilation: int = 1, 
        use_batchnorm: bool = True
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding,
                dilation = dilation,
                bias = not use_batchnorm
            ), 
            nn.ReLU(inplace = True), 
            nn.BatchNorm1d(num_features = out_channels) if use_batchnorm else nn.Identity() 
        )  
    
    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)    
    
    
class ConvTranspose2dBlock(nn.Module):

    def __init__(
        self, 
        in_channels: int,
        out_channels: int, 
        kernel_size: Union[int, Tuple[int, int]], 
        padding: Union[int, Tuple[int, int]] = 0, 
        output_padding: Union[int, Tuple[int, int]] = 0, 
        stride: Union[int, Tuple[int, int]] = 1, 
        dilation: Union[int, Tuple[int, int]] = 1, 
        use_batchnorm: bool = True,
        separable: bool = False 
    ) -> None:
        super().__init__()
        
        if not separable:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = padding,
                    output_padding = output_padding,
                    dilation = dilation,
                    bias = not use_batchnorm
                ), 
                nn.ReLU(inplace = True), 
                nn.BatchNorm2d(num_features = out_channels) if use_batchnorm else nn.Identity() 
            )  
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels = in_channels,
                    out_channels = in_channels,
                    kernel_size = kernel_size,
                    padding = padding,
                    stride = stride,
                    dilation = dilation,
                    groups = in_channels,
                    bias = False 
                ),
                nn.Conv2d(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = 1,
                    bias = not use_batchnorm
                ),
                nn.ReLU(inplace = True), 
                nn.BatchNorm2d(num_features = out_channels) if use_batchnorm else nn.Identity() 
            )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


    
class PoseDecoder(nn.Module):
    
    """
        Improvements over F1
        - Merge high res channels to allow more details.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = 256,
        highres_channels: int = 2048,
        mid_blocks: int = 4,
        heatmap_size: int = 112,
        soft_argmax_alpha: float = 1.0
    ) -> None:
        super().__init__()
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.mid_blocks = mid_blocks
        self.heatmap_size = heatmap_size
        self.soft_argmax_alpha = soft_argmax_alpha
        
        self.xy_block = nn.Sequential(*[
            ConvTranspose2dBlock(
                in_channels = in_channels if i == 0 else mid_channels,
                out_channels = mid_channels,
                # Parameters to upscale resolution by 2x.
                kernel_size = 4,
                stride = 2,
                padding = 1,
                separable = False if i == 0 else True 
            )
            for i in range(mid_blocks)
        ])
        
        self.xy_block_highres = ConvTranspose2dBlock(
            in_channels = mid_channels + highres_channels,
            out_channels = mid_channels,
            kernel_size = 4,
            stride = 2,
            padding = 1,
            separable = True 
        )
        
        self.z_block = nn.Sequential(*[
            ConvTranspose2dBlock(
                in_channels = in_channels if i == 0 else mid_channels,
                out_channels = mid_channels,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                separable = True 
            )
            for i in range(mid_blocks)
        ])
        
        self.z_block_highres = ConvTranspose2dBlock(
            in_channels = mid_channels + highres_channels,
            out_channels = mid_channels,
            kernel_size = 4,
            stride = 2,
            padding = 1,
            separable = True 
        )
        
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
        
        
    def forward(self, lowres: Tensor, highres: Tensor) -> Tensor: 
        """
            In shape: (B, I, S, S).
            · B: batch_size.
            · I: in_channels.
            · S: encoder output resolution. 
            Out shape: (B, O, 3).
            · O: out_channels (number of keypoints).
        """
        # Evaluate x, y axis heatmap features 
        xy_block = self.xy_block(lowres)
        xy_block = torch.cat([xy_block, highres], dim=1)
        xy_block = self.xy_block_highres(xy_block)
        
        z_block = self.z_block(lowres)
        z_block = torch.cat([z_block, highres], dim=1)
        z_block = self.z_block_highres(z_block)
        
        x = xy_block.mean(dim=(2))
        x = self.x_layer(x)
        x, x_heatmap = self.soft_argmax_1d(x)
        
        y = xy_block.mean(dim=(3))
        y = self.y_layer(y)
        y, y_heatmap= self.soft_argmax_1d(y)
        
        # Evaluate z axis (depth) heatmap features         
        z = z_block.mean(dim=(2))
        z = self.z_layer(z) 
        z, z_heatmap = self.soft_argmax_1d(z)
        
        xyz = torch.cat((x, y, z), dim=2)
        xyz_heatmap = torch.cat((x_heatmap, y_heatmap, z_heatmap), dim=2)
        xyz_heatmap = xyz_heatmap.view(-1, self.out_channels, 3, self.heatmap_size)
        return xyz, xyz_heatmap
        
        
    def soft_argmax_1d(self, x):
        # Differentiable argmax function 
        B, C, L = x.shape
        x = F.softmax(x * self.soft_argmax_alpha, dim=2)
        coord = x * torch.arange(start=0, end=self.heatmap_size).to(x.device)
        coord = coord.sum(dim=2, keepdim=True)
        return coord, x
    
    