import pathlib

import cv2
import matplotlib.pyplot as plt

from lib.processor import ImageProcessor


def show_images_non_block(images: list[ImageProcessor]):
    for i, image in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(cv2.cvtColor(image.value, cv2.COLOR_BGRA2RGBA))
        plt.axis("off")
    plt.show(block=False)
    return plt.close


def change_base_dir(base: pathlib.Path, path: str, suffix: str) -> pathlib.Path:
    new = pathlib.Path(path) / base.relative_to("input").with_suffix(suffix)
    new.parent.mkdir(parents=True, exist_ok=True)
    return new


def get_area(x: int, y: int, w: int, h: int) -> int:
    return (w - x) * (h - y)


def get_resize(x: int, y: int, size: int) -> tuple[int, int, int, int]:
    return -size, -size, x + size, y + size


def get_resize2(x: int, y: int, w: int, h: int, pad: int) -> tuple[int, int, int, int]:
    return x - pad, y - pad, w + pad, h + pad
