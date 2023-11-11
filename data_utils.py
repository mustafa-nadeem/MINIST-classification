"""Module used for unzip .gz files."""
import gzip
import struct
from typing import List
import numpy as np


def load_images(path: str) -> List[np.ndarray]:
    images = []
    with gzip.open(path, "rb") as file:
        header = struct.unpack(">4i", file.read(16))
        _, size, width, height = header

        chunk = width * height
        for _ in range(size):
            img = struct.unpack(">%dB" % chunk, file.read(chunk))
            img_np = np.array(img, np.uint8)
            images.append(img_np)

        return images


def load_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as file:
        header = struct.unpack(">2i", file.read(8))
        _, size = header

        labels = struct.unpack(">%dB" % size, file.read())

        return np.array(labels, np.int32)
