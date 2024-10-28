import glob
import json
import os
import shutil
from itertools import product

from lib.processor import ImageProcessor

OUTPUT_DIR = "output"


# 少数対応のrangeを作成
def frange(start, stop, step):
    while start < stop:
        yield round(start, 1)
        start += step


def get_name(i: int):
    return f"{i:03}" if i >= 0 else f"_{-i:03}"


if __name__ == "__main__":
    with open("input.json") as f:
        conf: dict = json.load(f)

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img = glob.glob(list(conf.values())[0])[0]
    data = ImageProcessor.from_path_without_base(img)

    matrix = [
        (-15, 15, 1),
        (0.5, 1.5, 0.1),
        (0.5, 1.5, 0.1),
    ]

    # matrix = [
    #     (-5, 5, 1),
    #     (0.9, 1.5, 0.1),
    #     (0.8, 1.5, 0.1),
    # ]
    os.makedirs(f"{OUTPUT_DIR}/product", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/1", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/2", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/3", exist_ok=True)

    for i in frange(*(matrix[0])):
        data.copy().hsv(i, 1, 1).write(f"{OUTPUT_DIR}/1/{get_name(i)}.png")

    for i in frange(*(matrix[1])):
        data.copy().hsv(0, i, 1).write(f"{OUTPUT_DIR}/2/{get_name(i)}.png")

    for i in frange(*(matrix[2])):
        data.copy().hsv(0, 1, i).write(f"{OUTPUT_DIR}/3/{get_name(i)}.png")

    pro = [matrix[0][0], matrix[0][1]], [matrix[1][0], matrix[1][1]], [matrix[2][0], matrix[2][1]]
    for i, ii, iii in product(*pro):
        data.copy().hsv(i, ii, iii).write(f"{OUTPUT_DIR}/product/{get_name(i)}_{get_name(ii)}_{get_name(iii)}.png")
