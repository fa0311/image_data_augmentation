import glob
import json
import os
import shutil

import cv2
import matplotlib.pyplot as plt

from lib.processor import ImageProcessor

OUTPUT_DIR = "output"


def frange(start, stop, step):
    while start < stop:
        yield round(start, 1)
        start += step


def frange2(count, start, end=None):
    end = end or abs(start)
    step = (end - start) / (count - 1)
    while count > 0:
        yield round(start, 1)
        start += step
        count -= 1


def get_name(i: int):
    return f"{i:03}" if i >= 0 else f"_{-i:03}"


if __name__ == "__main__":
    with open("input.json") as f:
        conf: dict = json.load(f)

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img = glob.glob(list(conf.values())[0]["input"])[0]
    data = ImageProcessor.from_path_without_base(img)

    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    for ax, iii in zip(axs, frange2(5, 0.5, 1.5)):
        for ax, ii in zip(ax, frange2(5, 0.5, 1.5)):
            img = cv2.cvtColor(data.copy().hsv(0, ii, iii).value, cv2.COLOR_BGRA2RGBA)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"h: 0 s: {ii} v: {iii}")
        fig.suptitle(f"hsv_{iii}.png")
        fig.tight_layout()
        fig.savefig(f"{OUTPUT_DIR}/hsv_{iii}.png")
