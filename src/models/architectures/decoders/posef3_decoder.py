from typing import List, Union, Tuple

import torch 
import torch.nn.functional as F
from torch import nn, Tensor


class Conv1dBlock(nn.Module):

    def __init__(
        self, 
        in_channels: int,
        out_channels: int, 
        kernel_size: int,
        stride: int = 1,  
        padding: int = 0, 
        dilation: int = 1, 
        groups: int = 1,
        padding_mode: str = 'zeros',
        use_activation: bool = True, 
        use_batchnorm: bool = True,
        use_separable: bool = False 
    ) -> None:
        super().__init__()
        
        block = []
        if not use_separable:
            block += [nn.Conv1d(
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
                nn.Conv1d(
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
            nn.BatchNorm1d(out_channels) if use_batchnorm else nn.Identity()  
        ]
        
        self.block = nn.Sequential(*block)
        
    
    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
    

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
    

class ConvTranspose2dBlock(nn.Module):

    def __init__(
        self, 
        in_channels: int,
        out_channels: int, 
        kernel_size: Union[int, Tuple[int, int]], 
        stride: Union[int, Tuple[int, int]] = 1,  
        padding: Union[int, Tuple[int, int]] = 0, 
        output_padding: Union[int, Tuple[int, int]] = 0,
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
            block += [nn.ConvTranspose2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding,
                output_padding = output_padding,
                dilation = dilation,
                groups = groups,
                padding_mode = padding_mode,
                bias = not use_batchnorm
            )] 
        else:
            block +=[
                nn.ConvTranspose2d(
                    in_channels = in_channels,
                    out_channels = in_channels,
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = padding,                    
                    output_padding = output_padding,
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


class AvgPool2dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            Conv2dBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=1
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        x = self.block(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return x


class Dilated2dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilations: List[int],
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        branches = []
        # 
        branches += [Conv2dBlock(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=1
        )]
        branches += [
            Conv2dBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                dilation=dilation,
                padding=dilation,
                use_separable=True
            )
            for dilation in dilations
        ]
        branches += [AvgPool2dBlock(
            in_channels=in_channels, 
            out_channels=out_channels)
        ]
        self.branches = nn.ModuleList(branches)

        self.head = nn.Sequential(
            Conv2dBlock(
                in_channels=(len(dilations) + 2) * out_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        res = []
        for branch in self.branches:
            res.append(branch(x))
        res = torch.cat(res, dim=1)
        return self.head(res)
    
    
class AxisDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 21,
        mid_channels: int = 256,
        mid_depth: int = 2,
        dilations: List[int] = [4, 8, 16],
        heatmap_size: int = 56,
        soft_argmax_alpha: float = 1.0
    ) -> None:
        super().__init__()
        self.heatmap_size = heatmap_size
        self.soft_argmax_alpha = soft_argmax_alpha
        
        # Computes axis features
        self.block_2d_axis = nn.Sequential(*[
            Dilated2dBlock(
                in_channels = mid_channels if i else in_channels,
                out_channels = mid_channels, 
                dilations = dilations
            ) for i in range(mid_depth)
        ])
        
        # Computes 1D-heatmap axis features
        self.block_1d_axis = nn.Sequential(*[
            Conv1dBlock(
                in_channels = mid_channels,
                out_channels = out_channels if i == mid_depth-1 else mid_channels,
                kernel_size = 1,
                stride = 1,
                padding = 0
            ) for i in range(mid_depth) 
        ])
        
    def forward(self, x: Tensor) -> Tensor: 
        x = self.block_2d_axis(x)
        x = x.mean(dim=(2))
        x = self.block_1d_axis(x)
        coord, heatmap = self.soft_argmax_1d(x)
        return coord, heatmap
        
    
    def soft_argmax_1d(self, x):
        # Differentiable argmax function 
        B, C, L = x.shape
        x = F.softmax(x * self.soft_argmax_alpha, dim=2)
        coord = x * torch.arange(start=0, end=self.heatmap_size).to(x.device)
        coord = coord.sum(dim=2, keepdim=True)
        return coord, x
    
class PoseF3Decoder(nn.Module):

    def __init__(
        self,
        in_highres_channels: int = 256,
        in_lowres_channels: int = 2048,
        low_to_high_upscales: int = 3,
        out_channels: int = 21,
        mid_channels: int = 128,
        mid_depth: int = 2,
        dilations: List[int] = [4, 8, 16],
        heatmap_size: int = 56
    ) -> None:
        super().__init__()
        self.heatmap_size = heatmap_size
        self.out_channels = out_channels
        
        # Computes highres features
        self.block_highres = nn.Sequential(*[
            Dilated2dBlock(
                in_channels = mid_channels if i else in_highres_channels,
                out_channels = mid_channels, 
                dilations = dilations
            ) for i in range(mid_depth)
        ])
        
        # Decodes lowres features 
        self.block_lowres = nn.Sequential(*[
            ConvTranspose2dBlock(
                in_channels = mid_channels if i else in_lowres_channels,
                out_channels = mid_channels,
                # Parameters to upscale resolution by 2x.
                kernel_size = 4,
                stride = 2,
                padding = 1,
                use_separable = True 
            )
            for i in range(low_to_high_upscales)
        ])
        
        # Computes merged features
        self.block_merge = nn.Sequential(*[
            Dilated2dBlock(
                in_channels = mid_channels if i else mid_channels * 2,
                out_channels = mid_channels, 
                dilations = dilations
            ) for i in range(mid_depth)
        ])
        
        axis_params = {
            "in_channels": mid_channels,
            "out_channels": out_channels,
            "mid_channels": mid_channels,
            "mid_depth": mid_depth,
            "dilations": dilations,
            "heatmap_size": heatmap_size,
        }
        
        # Computes x-axis features 
        self.x_block = AxisDecoder(**axis_params)
        self.y_block = AxisDecoder(**axis_params)
        self.z_block = AxisDecoder(**axis_params)
        

    def forward(self, lowres: Tensor, highres: Tensor) -> Tensor: 
        
        highres_features = self.block_highres(highres)
        lowres_features = self.block_lowres(lowres)
        
        merged = torch.cat([highres_features, lowres_features], dim=1)
        merged_features = self.block_merge(merged)
        
        x, x_hm = self.x_block(merged_features)
        y, y_hm = self.y_block(merged_features)
        z, z_hm = self.z_block(merged_features)
        
        xyz = torch.cat((x, y, z), dim=2)
        xyz_hm = torch.cat((x_hm, y_hm, z_hm), dim=2)
        xyz_hm = xyz_hm.view(-1, self.out_channels, 3, self.heatmap_size)
        
        return xyz, xyz_hm