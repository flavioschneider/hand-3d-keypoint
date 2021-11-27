from typing import List, Union, Tuple

import torch 
from torch import nn, Tensor 

class ConvBlock(nn.Module):

    def __init__(
        self, 
        in_channels: int,
        out_channels: int, 
        kernel_size: Union[int, Tuple[int, int]], 
        stride: Union[int, Tuple[int, int]] = 1, 
        padding: Union[int, Tuple[int, int]] = 0, 
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        use_activation: bool = True, 
        use_batchnorm: bool = True
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride = stride,
                padding = padding,
                dilation = dilation,
                groups = groups, 
                bias = not use_batchnorm
            ), 
            nn.SiLU(inplace = True) if use_activation else nn.Identity(), 
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity() 
        )  
    
    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
    

def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SEBlock(nn.Module):
    
    def __init__(
        self,
        channels: int, 
        reduction: int = 4        
    ):
        super().__init__()
        
        self.block = nn.Sequential(
            # Squeeze part
            nn.AdaptiveAvgPool2d(output_size = (1, 1)),
            # Excitation part 
            nn.Conv2d(
                in_channels = channels,
                out_channels = make_divisible(channels // reduction, 8),
                kernel_size = 1
            ),
            nn.SiLU(inplace = True), 
            nn.Conv2d(
                in_channels = make_divisible(channels // reduction, 8),
                out_channels = channels,
                kernel_size = 1
            ),
            nn.Sigmoid()
        )  
    
    def forward(self, x: Tensor) -> Tensor:
        return x * self.block(x)
    


class MBConv(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expansion: int = 1,
        reduction: int = 4,
        use_se: bool = True,
        use_fused: bool = False 
    ):
        super().__init__()
        self.use_skip = (stride == 1) and \
                        (kernel_size == 3) and \
                        (in_channels == out_channels)
        
        mid_channels = in_channels * expansion
        
        if use_fused:
            block = [
                ConvBlock(
                    in_channels = in_channels,
                    out_channels = mid_channels,
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = 1
                ),
                SEBlock(
                    channels = mid_channels,
                    reduction = reduction 
                ) if use_se else nn.Identity(),
                ConvBlock(
                    in_channels = mid_channels,
                    out_channels = out_channels,
                    kernel_size = 1,
                    use_activation = False 
                )
            ]
        else:
            block = [
                ConvBlock(
                    in_channels = in_channels,
                    out_channels = mid_channels,
                    kernel_size = 1
                ) if expansion != 1 else nn.Identity(),
                ConvBlock(
                    in_channels = mid_channels,
                    out_channels = mid_channels,
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = 1,
                    groups = mid_channels
                ),
                SEBlock(
                    channels = mid_channels,
                    reduction = reduction 
                ) if use_se else nn.Identity(),
                ConvBlock(
                    in_channels = mid_channels,
                    out_channels = out_channels,
                    kernel_size = 1,
                    use_activation = False 
                )
            ]
        
        self.block = nn.Sequential(*block)
        
    def forward(self, x: Tensor) -> Tensor:
        if self.use_skip:
            return x + self.block(x)
        else:
            return self.block(x)



class MBConvBlock(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layers: int,
        kernel_size: int = 3,
        stride: int = 1,
        expansion: int = 1,
        reduction: int = 4,
        use_se: bool = True,
        use_fused: bool = False 
    ):
        super().__init__() 
        
        block = []
        for i in range(layers):
            block += [MBConv(
                in_channels = in_channels if i == 0 else out_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                stride = stride if i == 0 else 1,
                expansion = expansion,
                reduction = reduction,
                use_se = use_se,
                use_fused = use_fused
            )]
        self.block = nn.Sequential(*block)
        
        
    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)
    


class EffNetV2Encoder(nn.Module):
    
    def __init__(
        self,
        name: str = 'effnetv2_s',
    ):
        super().__init__() 
        args = self.models[name] 
        self.in_channels = args["in_channels"] 
        self.out_channels = args["out_channels"]
        prev_channels = args["prev_channels"]
        params = args["params"]
            
        blocks = [
            nn.Identity(),
            ConvBlock(
                in_channels = self.in_channels,
                out_channels = prev_channels,
                kernel_size = 3,
                stride = 2,
                padding = 1
            )
        ]
        
        for expansion, channels, layers, stride, use_se, use_fused in params:
            blocks += [MBConvBlock(
                in_channels = prev_channels,
                out_channels = channels,
                layers = layers,
                stride = stride,
                expansion = expansion,
                use_se = use_se,
                use_fused = use_fused
            )]
            prev_channels = channels 
            
        self.blocks = nn.ModuleList(blocks)

        
    def forward(self, x: Tensor) -> List[Tensor]:
        outputs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outputs.append(x)
        return outputs
    

    models = {
        "effnetv2_s": {
            "in_channels": 3,   
            "prev_channels": 24, 
            "out_channels": [3, 24, 24, 48, 64, 128, 160, 256],   
            "params": [
                # [expansion, channels, layers, stride, use_se, use_fused]
                [1,  24,  2, 1, 0, 1],
                [4,  48,  4, 2, 0, 1],
                [4,  64,  4, 2, 0, 1],
                [4, 128,  6, 2, 1, 0],
                [6, 160,  9, 1, 1, 0],
                [6, 256, 15, 2, 1, 0],
            ]
        },
        "effnetv2_m": {
            "in_channels": 3,   
            "prev_channels": 24, 
            "out_channels": [3, 24, 24, 48, 80, 160, 176, 304, 512],   
            "params": [
                # [expansion, channels, layers, stride, use_se, use_fused]
                [1,  24,  3, 1, 0, 1],
                [4,  48,  5, 2, 0, 1],
                [4,  80,  5, 2, 0, 1],
                [4, 160,  7, 2, 1, 0],
                [6, 176, 14, 1, 1, 0],
                [6, 304, 18, 2, 1, 0],
                [6, 512,  5, 1, 1, 0],
            ]
        },
        "effnetv2_l": {
            "in_channels": 3,   
            "prev_channels": 24, 
            "out_channels": [3, 24, 32, 64, 96, 192, 224, 284, 640],   
            "params": [
                # [expansion, channels, layers, stride, use_se, use_fused]
                [1,  32,  4, 1, 0, 1],
                [4,  64,  7, 2, 0, 1],
                [4,  96,  7, 2, 0, 1],
                [4, 192, 10, 2, 1, 0],
                [6, 224, 19, 1, 1, 0],
                [6, 384, 25, 2, 1, 0],
                [6, 640,  7, 1, 1, 0],
            ]
        }
    }