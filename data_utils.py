"""Module used for unzip .gz files."""
import gzip
import struct
from typing import List

import numpy as np

def load_images(path: str) -> List[np.ndarray]:
    """Parses a MINIST images file and returns a list of numpy depicting the pixels"""
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
    """Parses a MINIST labels file and numpy array of the labels"""
    with gzip.open(path, "rb") as file:
        header = struct.unpack(">2i", file.read(8))
        _, size = header

        labels = struct.unpack(">%dB" % size, file.read())

        return np.array(labels, np.int32)
    

print(load_images("t10k-images-idx3-ubyte.gz"))