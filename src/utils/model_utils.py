import numpy as np
import torch 
from scipy.spatial.transform import Rotation as R
from torchvision.transforms import functional as F
from src.datamodules.transforms.transforms import h_flip_points



def heatmaps_gaussian_1D(size: int, means: torch.Tensor, std: float) -> torch.Tensor:
    """
        Produces 1D gaussian heatmap vectors of chosen size and std, one for each mean point. 

        size: int, size (S) of the heatmap
        means: Tensor, mean points to evaluate. Allowed shapes:
        - (B, N, D), (N, D), (N) for batch size B, number of points N, dimension D
        - e.g. (B, 21, 3)
        output: tensor heatamps. Shapes:
        - (B, N, D, S), (N, D, S), (N, S)
        - e.g. (B, 21, 3, 112)
    """
    shape = means.shape 
    means = means.view(-1)
    length = means.shape[0]
    gridx = torch.arange(size).to(means.device)
    gridx = gridx.repeat(length, 1).transpose(0, 1)
    heatmaps = torch.exp(-((gridx - means) ** 2) / 2.0 / std ** 2)
    heatmaps = heatmaps.transpose(0, 1).view(*shape, size)
    return heatmaps

def rotate_points(points: torch.Tensor, degrees: float):
    radians = np.radians(-degrees)
    rotation_vector = np.array([0, 0, radians])
    rotation = R.from_rotvec(rotation_vector)
    points = rotation.apply(points)
    return torch.tensor(points)

def rotate_image(img, degrees):
    return F.rotate(img, degrees, F.InterpolationMode.NEAREST, expand=False, center=None, fill=None)

def rotate_mirrored_points(points: torch.Tensor, degrees: float):
    return rotate_points(h_flip_points(points), degrees) 

def rotate_mirrored_image(img, degrees):
    return rotate_image(F.hflip(img), degrees) 
