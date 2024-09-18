import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import re
from functions import *
import pdb


class StreetViewDataset(Dataset):
    def __init__(
        self,
        root_dir,
        grid_lat,
        grid_lon,
        target_sigma,
        transform=None,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.target_sigma = target_sigma
        self.image_paths = [f for f in os.listdir(root_dir) if f.endswith(".jpg")]
        self.grid_lat = grid_lat
        self.grid_lon = grid_lon

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image, target, _, _ = self.__getitem_with_lat_lon__(idx)
        return image, target

    def __getitem_with_lat_lon__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(image_path).convert("RGB")

        lat, lon = self._parse_lat_lon_from_filename(self.image_paths[idx])

        if self.transform:
            image = self.transform(image)

        target = self._create_gaussian_target(lat, lon)

        return image, target, lat, lon

    def _parse_lat_lon_from_filename(self, filename):
        match = re.match(r"([-+]?\d*\.?\d+),([-+]?\d*\.?\d+).jpg", filename)
        if match:
            lat, lon = float(match.group(1)), float(match.group(2))
            return lat, lon
        else:
            raise ValueError(
                f"Filename {filename} does not match the pattern '<lat>,<lon>.jpg'"
            )

    def _create_gaussian_target(self, lat, lon):
        """Create a Gaussian target distribution centered on the true lat/lon."""
        distances = np.array(
            [
                haversine(lat, lon, self.grid_lat[i], self.grid_lon[i])
                for i in range(len(self.grid_lat))
            ]
        )
        target_distribution = np.exp(-(distances**2) / (2 * self.target_sigma**2))
        # IS THIS NECESSARY
        # target_distribution /= np.sum(target_distribution)  # Normalize to sum to 1
        return torch.tensor(target_distribution, dtype=torch.float32)
