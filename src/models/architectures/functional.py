import torch 
from torch import Tensor, nn 
from torch.nn import functional as F

class SoftArgmax2d(nn.Module):
    """
    Adapted from: 
    https://github.com/Ttayu/softargmax/blob/master/softargmax.py
    """
    
    def __init__(self, beta: int = 100, return_xy: bool = False):
        super().__init__()
        self.beta = beta
        self.return_xy = return_xy

    def forward(self, heatmap: Tensor) -> Tensor:
        heatmap = heatmap.mul(self.beta)
        batch_size, num_channel, height, width = heatmap.size()
        device: str = heatmap.device

        softmax: torch.Tensor = F.softmax(
            heatmap.view(batch_size, num_channel, height * width), dim=2
        ).view(batch_size, num_channel, height, width)

        xx, yy = torch.meshgrid(list(map(torch.arange, [width, height])))

        approx_x = (
            softmax.mul(xx.float().to(device))
            .view(batch_size, num_channel, height * width)
            .sum(2)
            .unsqueeze(2)
        )
        approx_y = (
            softmax.mul(yy.float().to(device))
            .view(batch_size, num_channel, height * width)
            .sum(2)
            .unsqueeze(2)
        )

        output = [approx_x, approx_y] if self.return_xy else [approx_y, approx_x]
        output = torch.cat(output, 2)
        return output


def soft_argmax2d(heatmap: Tensor, beta: int = 100, return_xy: bool = False) -> Tensor:
    return SoftArgmax2d(beta, return_xy)(heatmap)