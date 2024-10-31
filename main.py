import glob
import json
import os
import random
import shutil
from pathlib import Path

from tqdm import tqdm

from lib.annotate import annotate
from lib.processor import ImageProcessor

OUTPUT_DIR = "output"
OUTPUT_EXT = ".jpg"

OUTPUT_JPEG = f"{OUTPUT_DIR}/JPEGImages"
OUTPUT_ANNOTATION = f"{OUTPUT_DIR}/Annotations"
OUTPUT_IMAGESET = f"{OUTPUT_DIR}/ImageSets/Main"
OUTPUT_IGNORE = f"{OUTPUT_DIR}/ignore"
OUTPUT_LABELS = f"{OUTPUT_DIR}/labels.txt"
IMAGESET_FILES = ["test.txt", "train.txt", "trainval.txt", "val.txt"]


def data_augmentation(base: ImageProcessor) -> list[ImageProcessor]:
    res = []
    for i in range(4):
        data = base.copy()
        data.rotate(random.randint(0, 360))
        data.flip(random.randint(-1, 1))
        data.hsv(random.randint(-5, 5), random.uniform(0.9, 1.5), random.uniform(0.8, 1.5))

        res.append(data.copy())
    return res


def ramdom_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)


def flatten(data):
    return [item for sublist in data for item in sublist]


if __name__ == "__main__":
    with open("input.json") as f:
        conf: dict = json.load(f)

    ignore = {label: random.sample(glob.glob(path), 5) for label, path in conf.items()}
    ignore_flatten = flatten(ignore.values())

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_JPEG, exist_ok=True)
    os.makedirs(OUTPUT_ANNOTATION, exist_ok=True)
    os.makedirs(OUTPUT_IMAGESET, exist_ok=True)
    os.makedirs(OUTPUT_IGNORE, exist_ok=True)

    for label, path in tqdm(conf.items(), desc="Processing", leave=False):
        for file in tqdm(glob.glob(path)[:5], desc=label, leave=False):
            if file not in ignore_flatten:
                filename, ext = os.path.splitext(os.path.basename(file))
                base_data = ImageProcessor.from_path_without_base(file)
                augmentation = data_augmentation(base_data)
                for i, data in enumerate(augmentation):
                    data.resize_axis_x(512).set_base_color(ramdom_color())
                    border_size, size = data.get_border_size(), data.get_size()
                    data.paste().write(f"{OUTPUT_JPEG}/{filename}_{i}{OUTPUT_EXT}")
                    path_anno = f"{OUTPUT_ANNOTATION}/{filename}_{i}{OUTPUT_EXT}"
                    annotated = annotate(Path(path_anno), border_size, size, label)
                    annotated.write(f"{OUTPUT_ANNOTATION}/{filename}_{i}.xml")

    for dir in ignore.values():
        for file in dir:
            filename, ext = os.path.splitext(os.path.basename(file))
            from_path = Path(file)
            to_path = Path(f"{OUTPUT_IGNORE}/{filename}{OUTPUT_EXT}")
            data = ImageProcessor.from_path_without_base(str(from_path))
            data.copy().resize_axis_x(512).write(str(to_path))

    with open(OUTPUT_LABELS, "w") as f:
        for label in conf.keys():
            f.write(f"{label}\n")

    for path_name in IMAGESET_FILES:
        path = Path(f"{OUTPUT_IMAGESET}/{path_name}")
        path.touch(exist_ok=True)
        with open(path, "w") as f:
            for file in glob.glob(f"{OUTPUT_JPEG}/*{OUTPUT_EXT}"):
                f.write(f"{Path(file).stem}\n")
