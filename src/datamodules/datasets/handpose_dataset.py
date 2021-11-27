from typing import Any, Optional, Tuple, Callable

import os
import json
import torch
import numpy as np
from PIL import Image
from src.utils.data_utils import Hand


class HandposeDataset(torch.utils.data.Dataset):

    train_images_dirs = ["train", "val"]
    train_files_camera = ["train_K.json", "val_K.json"]
    train_files_xyz = ["train_xyz.json", "val_xyz.json"]

    test_images_dirs = ["test"]
    test_files_camera = ["test_K.json"]

    def __init__(
        self,
        root: str = "data/",
        train: bool = True,
        transforms: Optional[Callable] = None,
        transforms_pair: Optional[Callable] = None,
        limit: int = None,
    ) -> None:
        self.root = root
        self.train = train
        self.transforms = transforms
        self.transforms_pair = transforms_pair
        self.limit = limit

        if not self._check_exists():
            raise RuntimeError(f"Dataset (train={train}) not found in root '{root}'.")

        self.x_images: List[str] = []
        self.x_cameras: Optional[np.ndarray] = None  # (N, 3, 3)
        self.y_points: Optional[np.ndarray] = None  # (N, 21, 3)

        self._preprocess()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        x_image = Image.open(self.x_images[index]).convert("RGB")
        x_camera = torch.tensor(self.x_cameras[index])

        if self.train:
            y_points = torch.tensor(self.y_points[index])

            if self.transforms_pair is not None:
                # These transformations need to be done before x_image is converted to tensor
                x_image, y_points, _ = self.transforms_pair(x_image, y_points, x_camera)

            if self.transforms is not None:
                x_image = self.transforms(x_image)

            return x_image, x_camera, y_points

        else:
            if self.transforms is not None:
                x_image = self.transforms(x_image)

            return x_image, x_camera

    def __len__(self) -> int:
        return self.limit if self.limit is not None else len(self.x_images)

    def _check_exists(self) -> bool:
        # Checks that all files necessary are available
        if self.train:
            paths = self.train_images_dirs + self.train_files_camera + self.train_files_xyz
        else:
            paths = self.test_images_dirs + self.test_files_camera

        return all(os.path.exists(os.path.join(self.root, path)) for path in paths)

    def _preprocess(self) -> None:

        print("Preprocessing handpose dataset...")

        if self.train:
            x_images_paths = [os.path.join(self.root, folder) for folder in self.train_images_dirs]
            x_cameras_paths = [os.path.join(self.root, file) for file in self.train_files_camera]
            y_points_paths = [os.path.join(self.root, file) for file in self.train_files_xyz]
        else:
            x_images_paths = [os.path.join(self.root, folder) for folder in self.test_images_dirs]
            x_cameras_paths = [os.path.join(self.root, file) for file in self.test_files_camera]
            y_points_paths = []

        # Save all input image paths (we will lazily load images on __getitem__)
        for folder in x_images_paths:
            self.x_images += [
                os.path.join(folder, file) for file in os.listdir(folder) if os.path.isfile(os.path.join(folder, file))
            ]
        self.x_images = np.sort(self.x_images)
        # Save all input camera tensors
        x_cameras_list = [np.tile(self._json_to_array(path), (4, 1, 1)) for path in x_cameras_paths]
        self.x_cameras = np.concatenate(x_cameras_list, axis=0)
        # Save all output points tensors in kp3d format
        y_points_list = [
            np.tile(self._xyz_to_kp3d_points(self._json_to_array(path)), (4, 1, 1)) for path in y_points_paths
        ]
        self.y_points = None if not self.train else np.concatenate(y_points_list, axis=0)

        print(f"Done, preprocessed {self.__len__()} data points (train={self.train}).")

    def _json_to_array(self, dir) -> np.ndarray:
        # Reads the content of a json file into a numpy array
        with open(dir, "r") as file:
            content = json.load(file)
        return np.array(content)

    def _xyz_to_kp3d_points(self, xyz_points):
        # Convert order from FreiHAND (xyz) to AIT (kp3d). Accepts batch and sample input.
        output = np.zeros(shape=xyz_points.shape, dtype=xyz_points.dtype)

        output[..., Hand.root, :] = xyz_points[..., 0, :]
        output[..., Hand.thumb_mcp, :] = xyz_points[..., 1, :]
        output[..., Hand.thumb_pip, :] = xyz_points[..., 2, :]
        output[..., Hand.thumb_dip, :] = xyz_points[..., 3, :]
        output[..., Hand.thumb_tip, :] = xyz_points[..., 4, :]

        output[..., Hand.index_mcp, :] = xyz_points[..., 5, :]
        output[..., Hand.index_pip, :] = xyz_points[..., 6, :]
        output[..., Hand.index_dip, :] = xyz_points[..., 7, :]
        output[..., Hand.index_tip, :] = xyz_points[..., 8, :]

        output[..., Hand.middle_mcp, :] = xyz_points[..., 9, :]
        output[..., Hand.middle_pip, :] = xyz_points[..., 10, :]
        output[..., Hand.middle_dip, :] = xyz_points[..., 11, :]
        output[..., Hand.middle_tip, :] = xyz_points[..., 12, :]

        output[..., Hand.ring_mcp, :] = xyz_points[..., 13, :]
        output[..., Hand.ring_pip, :] = xyz_points[..., 14, :]
        output[..., Hand.ring_dip, :] = xyz_points[..., 15, :]
        output[..., Hand.ring_tip, :] = xyz_points[..., 16, :]

        output[..., Hand.pinky_mcp, :] = xyz_points[..., 17, :]
        output[..., Hand.pinky_pip, :] = xyz_points[..., 18, :]
        output[..., Hand.pinky_dip, :] = xyz_points[..., 19, :]
        output[..., Hand.pinky_tip, :] = xyz_points[..., 20, :]
        return output * 1000
