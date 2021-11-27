from typing import Any, List, Callable

from torchvision import transforms
from torchvision.transforms import functional as F
from scipy.spatial.transform import Rotation as R
import random
import torch
import numpy as np


def h_flip_points(points):
    points = points.clone()
    points[:, 0] = points[:, 0] * -1
    return points


def v_flip_points(points):
    points = points.clone()
    points[:, 1] = points[:, 1] * -1
    return points


def points_3d_to_2d(points, cam):
    f_x = cam[0, 0]
    f_y = cam[1, 1]
    o_x = cam[0, 2]
    o_y = cam[1, 2]

    points_2d = points.clone()
    points_2d[:, 0] = f_x * points_2d[:, 0] / points[:, 2] + o_x
    points_2d[:, 1] = f_y * points_2d[:, 1] / points[:, 2] + o_y

    return points_2d


def points_2d_to_3d(points, cam):
    f_x = cam[0, 0]
    f_y = cam[1, 1]
    o_x = cam[0, 2]
    o_y = cam[1, 2]

    points_3d = points.clone()
    points_3d[:, 0] = (points_3d[:, 0] - o_x) * points[:, 2] / f_x
    points_3d[:, 1] = (points_3d[:, 1] - o_y) * points[:, 2] / f_y

    return points_3d


def transform_points(points, T, cam):
    points_2d = points_3d_to_2d(points, cam)
    points_2d[:, 2] = 1
    transformed = points_2d @ T.T
    transformed[:, 2] = points[:, 2]
    points_3d = points_2d_to_3d(transformed, cam)
    return points_3d


def transform_image(img, angle=0, translate=(0, 0), scale=1):
    transformed = F.affine(img, angle, translate, scale, 0, interpolation=F.InterpolationMode.BILINEAR)
    return transformed


class RandomHorizontalFlip:
    """
    Randomly flips image and 3d points horizontally.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img, points, cam) -> Any:
        if torch.rand(1) > self.p:
            img = F.hflip(img)
            points = h_flip_points(points)

        return img, points, cam


class RandomRotation:
    """
    Randomly rotates image and 3d points in range (-degrees, +degrees)
    """

    def __init__(
        self, degrees, interpolation=F.InterpolationMode.NEAREST, expand=False, center=None, fill=None
    ) -> None:
        if degrees < 0:
            raise ValueError("Degrees must be positive.")
        self.degrees = (-degrees, degrees)

        self.interpolation = interpolation  # default is nearest, good for labels
        self.expand = expand
        self.center = center
        self.fill = fill

    def __call__(self, img, points, cam) -> Any:
        angle = random.uniform(self.degrees[0], self.degrees[1])
        radians = np.radians(-1 * angle)
        rotation_vector = np.array([0, 0, radians])
        rotation = R.from_rotvec(rotation_vector)

        points = torch.tensor(rotation.apply(points))
        img = F.rotate(img, angle, self.interpolation, self.expand, self.center, self.fill)

        return img, points, cam


class RandomVerticalFlip:
    """
    Randomly flips image and 3d points horizontally.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img, points, cam) -> Any:
        if torch.rand(1) > self.p:
            img = F.vflip(img)
            points = v_flip_points(points)

        return img, points, cam


class RandomScale:
    def __init__(self, scale_range=(0.8, 1.2)) -> None:
        self.scale_range = scale_range

    def __call__(self, img, points, cam) -> Any:
        factor = np.random.uniform(*self.scale_range)
        shift = np.array(img.size) / 2

        # Shift center to (0,0), scale, shift back
        T = (
            np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]], dtype=np.float64)
            @ np.array([[factor, 0, 0], [0, factor, 0], [0, 0, 1]])
            @ np.array([[1, 0, -shift[0]], [0, 1, -shift[1]], [0, 0, 1]], dtype=np.float64)
        )

        img = transform_image(img, scale=factor)
        points = transform_points(points, T, cam)

        return img, points, cam


class RandomTranslation:
    """
    Shifts image and points horizontally and vertically.
    Horizontal shift is by pixel value chosen uniformly from [-horizontal, horizontal].
    Vertical shift is by pixel value chosen uniformly from [-vertical, vertical].
    """

    def __init__(self, horizontal=10, vertical=10) -> None:
        self.horizontal = horizontal
        self.vertical = vertical

    def __call__(self, img, points, cam) -> Any:

        T = np.array([[1, 0, self.horizontal], [0, 1, self.vertical], [0, 0, 1]], dtype=np.float64)

        img = transform_image(img, translate=(self.horizontal, self.vertical))
        points = transform_points(points, T, cam)

        return img, points, cam


class Compose:
    """
    Composes several transforms together.
    """

    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, *inputs: Any) -> Any:
        for transform in self.transforms:
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
            inputs = transform(*inputs)
        return inputs


class ToTensor:
    """
    Converts all inputs to tensors.
    """

    def __init__(self) -> None:
        self.transform = transforms.ToTensor()

    def __call__(self, *inputs: Any) -> Any:
        outputs = list(inputs)

        for index, _input in enumerate(outputs):
            outputs[index] = self.transform(_input)

        return outputs


class ColorJitter:
    """
    Applies random color jitter to image
    """

    def __init__(self, brightness: float = 0, contrast: float = 0, saturation: float = 0) -> None:
        self.transform = transforms.ColorJitter(brightness, contrast, saturation)

    def __call__(self, image: Any, points: Any, cam: Any) -> Any:
        image = self.transform(image)
        return image, points, cam


class RandomChoiceCompose:
    """
    Randomly choose to apply one transform from a collection of transforms.
    """

    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, *inputs: Any) -> Any:
        transform = random.choice(self.transforms)
        outputs = transform(*inputs)
        return outputs

class Identity:
    """
    Doesn't apply any transform, useful e.g. in random choice to keep the original image sometimes
    """
    def __init__(self) -> None:
        pass

    def __call__(self, *inputs: Any) -> Any:
        return inputs

