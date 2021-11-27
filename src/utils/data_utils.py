from typing import TypeVar, Sequence, List, Optional

import torch 
import numpy as np
import torchvision 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch import Tensor 
from torch.utils.data.dataset import Dataset, Subset
from torch._utils import _accumulate
T = TypeVar('T')

class Hand(object):
    """
    Definition of joint ordering
    """
    root = 0
    thumb_mcp = 1
    index_mcp = 2
    middle_mcp = 3
    ring_mcp = 4
    pinky_mcp = 5
    thumb_pip = 6
    index_pip = 7
    middle_pip = 8
    ring_pip = 9
    pinky_pip = 10
    thumb_dip = 11
    index_dip = 12
    middle_dip = 13
    ring_dip = 14
    pinky_dip = 15
    thumb_tip = 16
    index_tip = 17
    middle_tip = 18
    ring_tip = 19
    pinky_tip = 20

    @classmethod
    def normalize_points(cls, points: Tensor) -> Tensor:
        """ 
        Normalizes a set of hand keypoints using the length of the index finger bone.
        points: (B, 21, 3)
        """
        bone_vectors = points[:, cls.index_mcp, :] - points[:,cls.index_pip,:]
        bone_lengths = torch.norm(bone_vectors, p=2, dim=1).repeat_interleave(21).view(-1, 21, 1)
        return points / bone_lengths

    @classmethod
    def to_2d_points(cls, points: Tensor, cameras: Tensor):
        """
        Converts a batch of 3D key-points to 2D using camera matrices. 
        points: (B, N, 3)
        cameras: (B, 3, 3)
        """
        B, N, _ = points.shape
        points_2d = torch.bmm(points, cameras.transpose(1, 2))
        points_2d = points_2d.view(-1, 3)
        points_2d = points_2d / points_2d[:, 2].view(-1, 1)
        points_2d = points_2d[:, 0:2]
        points_2d = points_2d.view(B, N, 2)
        return points_2d 

    @classmethod 
    def plot_2d(obj, images, points, points_secondary=None, cols=4, img_size=224):
        """
        Plots batch of images and 2D points. 
        images: (B, img_size, img_size, 3)
        points: (B, 21, 2)
        """
        B = images.shape[0]
        rows = B // cols
        to_pil = torchvision.transforms.ToPILImage()
        fig = make_subplots(
            rows=rows, 
            cols=cols, 
            shared_yaxes=True,
            shared_xaxes=True
        )
        
        for i in range(B):
            row = i // cols + 1
            col = i % cols + 1
            colors = [
                'rgba(246, 229, 141, 0.7)', 
                'rgba(255, 190, 118, 0.7)', 
                'rgba(255, 121, 121, 0.7)', 
                'rgba(224, 86, 253, 0.7)', 
                'rgba(126, 214, 223, 0.7)'
            ]
            
            for j in range(5):
                finger_points = [0]+list(range(j + 1, 21, 5))
                if points_secondary is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=points_secondary[i,finger_points,0], 
                            y=img_size - points_secondary[i,finger_points,1],
                            mode='lines+markers',
                            marker=dict(size=2.5, color='rgba(223, 249, 251, 0.7)'),
                            line=dict(width=2)
                        ), row, col
                    )
                fig.add_trace(
                    go.Scatter(
                        x=points[i,finger_points,0], 
                        y=img_size - points[i,finger_points,1],
                        mode='lines+markers',
                        marker=dict(size=2.5, color=colors[j]),
                        line=dict(width=2)
                    ), row, col
                )
                
            fig.add_layout_image(
                source=to_pil(images[i]),
                xref="x",
                yref="y",
                x=0,
                y=img_size,
                sizex=img_size,
                sizey=img_size,
                sizing="stretch",
                layer="below",
                row=row, 
                col=col
            )

        fig.update_layout(
            height=280*rows,
            showlegend=False
        )
            
        fig.update_xaxes(
            range=[0, img_size], # sets the range of xaxis
            constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
            gridcolor='rgba(255, 255, 255, 0.2)'
        )

        fig.update_yaxes(
            scaleanchor = "x",
            scaleratio = 1,
            gridcolor='rgba(255, 255, 255, 0.2)'
        )

        return fig 


class Scale:
    
    # Keypoints interval (x, y, z)
    KP = torch.Tensor([
        [-127.42727995, 121.29122019], 
        [-135.15593112, 129.37641144], 
        [297.07068205, 1040.35663605]
    ])
    # Keypoints normalized  (using Hand.normalize_points) interval
    KPN = torch.Tensor([
        [-5.5001, 4.9102], 
        [-5.7931, 4.5171], 
        [8.3581, 52.7773]
    ])
    # Points in 2D grid 
    def P2D(size=224): return torch.Tensor([0, size]).repeat(2, 1)
    # Points in 3D grid 
    def P3D(size): return torch.Tensor([0, size]).repeat(3, 1)
    # Heatmap interval with configurable size 
    def HM(size): return torch.Tensor([0, size]).repeat(3, 1)

    @staticmethod
    def linear(points: Tensor, domain, range) -> Tensor:
        """
            points: Tensor of shape (B, N, D) or (N, D) or (D)
            - B batch size
            - N number of points 
            - D dimension of points 
            domain: (2, D) min and max values of all dimensions
            range: (2, D)
            output: same as input but scaled.
        """
        # Put all tensors on same device as points.
        domain = domain.to(points.device)
        range = range.to(points.device)
        # Save shape and reshape points.
        shape = points.shape 
        dimension = domain.shape[0]
        points = points.view(-1, dimension)
        # Compute scaled points. 
        min_domain_values, max_domain_values = domain.transpose(0, 1)
        diff_domain_values = min_domain_values - max_domain_values 
        min_range_values, max_range_values = range.transpose(0, 1)
        diff_range_values = min_range_values - max_range_values 
        scaled_points = min_range_values + ((points - min_domain_values) / diff_domain_values) * diff_range_values
        # Return scaled points and restore shape
        return scaled_points.view(shape)

class Heatmap:

    @staticmethod
    def make_gaussians(
        means: Tensor, 
        size: int, 
        sigma: float
    ):
        """
        Given a batch/set of means it produces grids of gaussians, the dimension is
        picked from the last dimension of the means (i.e. works in 1D, 2D, 3D, ...)
        - means: (N*, D) e.g. (N, D) or (B, N, D).
        - output: (N*, [size]*D)
        """
        shape = means.shape[:-1]
        D = means.shape[-1]
        num_means = torch.prod(torch.tensor(shape))
        num_points = size ** D 
        means = means.view(-1, D) 
        # Find all points in grid (size num_points*D)
        axis = torch.arange(size).to(means.device)
        grid = torch.meshgrid([axis] * D) 
        grid = torch.cat(grid)
        points = grid.view(D, -1).transpose(0, 1)
        # Vectorized Gaussian between all points and all means.
        points = points.repeat(num_means, 1)
        means = means.repeat_interleave(num_points, dim=0)
        #print((points-means).shape)
        norms = ((points - means) ** 2).sum(dim=1)
        gaussians = torch.exp(-norms / 2.0 / sigma ** 2)
        gaussians = gaussians.view(list(shape)+[size]*D)
        return gaussians

    @staticmethod
    def plot_2d(
        heatmaps: Tensor, 
        heatmaps_secondary: Optional[Tensor] = None, 
        cols: int = 7
    ):
        """
        heatmaps: (N, S, S) N heatmaps, of size SxS. 
        heatmaps_pred: (N, S, S). 
        """
        N, S, S = heatmaps.shape 
        
        use_secondary = heatmaps_secondary is not None
        fig = make_subplots(
            rows=1+use_secondary, 
            cols=1, 
            shared_yaxes=False,
            shared_xaxes=False 
        )
        
        rows = N // cols
        heatmaps = heatmaps.view(rows, cols, S, S)
        heatmaps = heatmaps.permute(0,2,1,3).reshape(rows*S, cols*S)
        
        fig.add_trace(
            go.Heatmap(
                z=heatmaps,
                colorscale='jet'
            ), 1, 1
        )
        fig.update_yaxes(scaleanchor = "x", row=1, col=1)
        
        if use_secondary:
            heatmaps_secondary = heatmaps_secondary.view(rows, cols, S, S)
            heatmaps_secondary = heatmaps_secondary.permute(0,2,1,3).reshape(rows*S, cols*S)
            fig.add_trace(
                go.Heatmap(
                    z=heatmaps_secondary,
                    colorscale='jet'
                ), 2, 1
            )
            fig.update_yaxes(scaleanchor = "x", row=2, col=1)

        fig.update_layout(height=500+500*use_secondary)

        return fig


# 0,1,2,...,91167, 91168, 91168

def linear_split(
    dataset: Dataset[T], 
    lengths: Sequence[int]
) -> List[Subset[T]]:
    """
    Implementation of deterministic linear split that follows the same 
    conventions as torch.utils.data.dataset.random_split 
    """
    total_length = sum(lengths)
    if total_length != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    indices = torch.arange(total_length)
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]
    
def custom_split(
    dataset: Dataset[T], 
    lengths: Sequence[int]
) -> List[Subset[T]]:
    """
    Implementation of custom split that accounts for the fact that 
    the validation set has data augmentations in it and might leak data.
    """
    total_length = sum(lengths)
    if total_length != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    start, offset, blocks = 91168, 4884, 4
    indices = torch.arange(start)
    indices = torch.cat([indices, torch.tensor([start+i+offset*j for i in range(offset) for j in range(blocks)])])
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]
    


"""
gridx, gridy, gridz = np.mgrid[0:size, 0:size, 0:size]
D = (gridx - mean[0]) ** 2 + (gridy - mean[1]) ** 2 + (gridz - mean[2]) ** 2
return np.exp(-D / 2.0 / std**2)
"""