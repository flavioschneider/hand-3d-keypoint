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


  

class LitePoseDecoder(nn.Module):
    
    def __init__(
        self,
        in_highres_channels: int = 256,
        in_lowres_channels: int = 2048,
        low_to_high_upscales: int = 3,
        out_channels: int = 21,
        mid_channels: int = 128,
        mid_depth: int = 2,
        heatmap_size: int = 112,
        soft_argmax_alpha: float = 10.0 
    ) -> None:
        super().__init__()
        self.heatmap_size = heatmap_size
        self.out_channels = out_channels
        self.soft_argmax_alpha = soft_argmax_alpha
        
        self.block_lowres = nn.Sequential(*[
            ConvTranspose2dBlock(
                in_channels = mid_channels if i else in_lowres_channels,
                out_channels = mid_channels,
                # Parameters to upscale resolution by 2x.
                kernel_size = 4,
                stride = 2,
                padding = 1
            )
            for i in range(low_to_high_upscales)
        ])

        self.block_xy_highres = ConvTranspose2dBlock(
            in_channels = mid_channels + in_highres_channels,
            out_channels = mid_channels,
            kernel_size = 4,
            stride = 2,
            padding = 1
        )
        
        self.block_z_highres = ConvTranspose2dBlock(
            in_channels = mid_channels + in_highres_channels,
            out_channels = mid_channels,
            kernel_size = 4,
            stride = 2,
            padding = 1
        )
        
        self.x_layer = nn.Sequential(*[
            nn.Conv1d(
                in_channels = mid_channels,
                out_channels = mid_channels if i != mid_depth-1 else out_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ) for i in range(mid_depth)
        ]) 
        
        self.y_layer = nn.Sequential(*[
            nn.Conv1d(
                in_channels = mid_channels,
                out_channels = mid_channels if i != mid_depth-1 else out_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ) for i in range(mid_depth)
        ]) 
        
        self.z_layer = nn.Sequential(*[
            nn.Conv1d(
                in_channels = mid_channels,
                out_channels = mid_channels if i != mid_depth-1 else out_channels,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ) for i in range(mid_depth)
        ]) 
        
        
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
        lowres_features = self.block_lowres(lowres)
        highres = torch.cat([lowres_features, highres], dim=1)

        xy_highres_features = self.block_xy_highres(highres)
        z_highres_features = self.block_z_highres(highres)
        
        x = xy_highres_features.mean(dim=(2))
        x = self.x_layer(x)
        x, x_heatmap = self.soft_argmax_1d(x)
        
        y = xy_highres_features.mean(dim=(3))
        y = self.y_layer(y)
        y, y_heatmap = self.soft_argmax_1d(y)
        
        # Evaluate z axis (depth) heatmap features         
        z = z_highres_features.mean(dim=(2))
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
    
    