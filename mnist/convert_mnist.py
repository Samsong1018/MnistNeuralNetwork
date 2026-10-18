#!/usr/bin/env python3
"""Convert raw MNIST .gz files into numpy archives.

Writes:
  mnist/mnist_28.npz  original 28x28 images
  mnist/mnist_24.npz  resized to 24x24 to match digit_collector.py output

Each archive has x_train, y_train, x_test, y_test (uint8 images 0-255, uint8 labels).
"""

import gzip
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"


def read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    ndim = magic & 0xFF
    dims = [int.from_bytes(data[4 + 4 * i:8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def resize(images: np.ndarray, size: int) -> np.ndarray:
    return np.stack([
        np.asarray(Image.fromarray(img).resize((size, size), Image.LANCZOS))
        for img in images
    ])


def main() -> None:
    x_train = read_idx(RAW / "train-images-idx3-ubyte.gz")
    y_train = read_idx(RAW / "train-labels-idx1-ubyte.gz")
    x_test = read_idx(RAW / "t10k-images-idx3-ubyte.gz")
    y_test = read_idx(RAW / "t10k-labels-idx1-ubyte.gz")

    np.savez_compressed(HERE / "mnist_28.npz", x_train=x_train, y_train=y_train,
                        x_test=x_test, y_test=y_test)
    np.savez_compressed(HERE / "mnist_24.npz", x_train=resize(x_train, 24), y_train=y_train,
                        x_test=resize(x_test, 24), y_test=y_test)
    print(f"train {x_train.shape}, test {x_test.shape} -> mnist_28.npz, mnist_24.npz")


if __name__ == "__main__":
    main()
