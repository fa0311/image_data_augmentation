import glob
import json
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
from tqdm import tqdm

from lib.annotate import annotate
from lib.noise import NoiseImage
from lib.processor import ImageProcessor

OUTPUT_DIR = "output"
OUTPUT_EXT = ".jpg"
OUTPUT_SIZE = 64
OUTPUT_COUNT = 8
MAX_WORKERS = None
INCLUDE_BASE = False

OUTPUT_JPEG = f"{OUTPUT_DIR}/JPEGImages"
OUTPUT_ANNOTATION = f"{OUTPUT_DIR}/Annotations"
OUTPUT_IMAGESET = f"{OUTPUT_DIR}/ImageSets/Main"
OUTPUT_IGNORE = f"{OUTPUT_DIR}/ignore"
OUTPUT_LABELS = f"{OUTPUT_DIR}/labels.txt"
IMAGESET_TRAIN_FILES = f"{OUTPUT_IMAGESET}/trainval.txt"
IMAGESET_TEST_FILES = f"{OUTPUT_IMAGESET}/test.txt"


def data_augmentation(base: ImageProcessor, count: int) -> list[ImageProcessor]:
    res = []
    for i in range(count):
        data = base.copy()
        data.rotate(random.randint(0, 360))
        data.flip(random.randint(-1, 1))
        data.hsv(random.randint(-5, 5), random.uniform(0.8, 1.2), random.uniform(0.8, 1.2))
        res.append(data)
    return res


def random_resize(base: ImageProcessor, size: int, value_size: int) -> ImageProcessor:
    base.trim(*base.get_border_size())

    source_x, source_y = base.get_size()
    base.resize_axis_x(value_size) if source_x > source_y else base.resize_axis_y(value_size)

    new_x, new_y = base.get_size()
    top_margin, right_margin = random.randint(1, size - new_y - 1), random.randint(1, size - new_x - 1)
    res = base.add_margin((top_margin, right_margin, size - new_y - top_margin, size - new_x - right_margin))
    return res


def ramdom_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)


def flatten(data):
    return [item for sublist in data for item in sublist]


def process_file(args):
    file, label, orig = args
    filename, ext = os.path.splitext(os.path.basename(file))
    base_data = ImageProcessor.from_path_without_base(file)
    augmentation = data_augmentation(base_data, OUTPUT_COUNT)
    for i, data in enumerate(augmentation):
        randsize = random.randint(OUTPUT_SIZE // 4, (OUTPUT_SIZE - OUTPUT_SIZE // 4))
        data = random_resize(data.remove_noise(), OUTPUT_SIZE, randsize)
        rand = random.random()
        if rand < 0.1:
            data.set_base_color(ramdom_color())
        elif rand < 0.2:
            data.set_base_color((255, 255, 255, 255))
        else:
            noise = NoiseImage().generate(N=OUTPUT_SIZE, count=5)
            data.set_base(cv2.cvtColor(noise.astype("uint8"), cv2.COLOR_BGR2BGRA))
        annotate_file(data, label, f"{filename}_{i}")

    if  INCLUDE_BASE:
        base = ImageProcessor.from_path_base(file, str(orig))
        source_x, source_y = base.get_size()
        base.resize_axis_x(OUTPUT_SIZE, True) if source_x > source_y else base.resize_axis_y(OUTPUT_SIZE, True)
        base.square(base=True)
        annotate_file(base, label, f"{filename}_{OUTPUT_COUNT}")
        return [f"{filename}_{i}" for i in range(OUTPUT_COUNT + 1)]
    else:
        return [f"{filename}_{i}" for i in range(OUTPUT_COUNT)]


def annotate_file(data: ImageProcessor, label: str, filename: str):
    border_size, _ = data.get_border_size(), data.get_size()
    data.paste().write(f"{OUTPUT_JPEG}/{filename}{OUTPUT_EXT}")
    path_anno = f"{OUTPUT_ANNOTATION}/{filename}{OUTPUT_EXT}"
    annotated = annotate(Path(path_anno), border_size, (OUTPUT_SIZE, OUTPUT_SIZE), label)
    annotated.write(f"{OUTPUT_ANNOTATION}/{filename}.xml")


if __name__ == "__main__":
    with open("input.json") as f:
        load = json.load(f)

    input = {k: v["input"] for k, v in load["input"].items()}
    original = {k: v["original"] for k, v in load["input"].items()}

    ignore = {label: random.sample(glob.glob(path), 5) for label, path in input.items()}

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_JPEG, exist_ok=True)
    os.makedirs(OUTPUT_ANNOTATION, exist_ok=True)
    os.makedirs(OUTPUT_IMAGESET, exist_ok=True)
    os.makedirs(OUTPUT_IGNORE, exist_ok=True)

    originals = {label: [Path(x) for x in glob.glob(original[label])] for label in input.keys()}
    def get_orig(label, path):
        filename, ext = os.path.splitext(os.path.basename(path))
        return [x for x in originals[label] if x.stem.lower() == filename.lower()][0]

    file_list = [[(x, label,get_orig(label,x)) for x in glob.glob(path) if x not in ignore[label]] for label, path in input.items()]
    files = flatten(file_list)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        output = list(tqdm(executor.map(process_file, files), total=len(files), desc="Files", leave=False))

    Path(IMAGESET_TRAIN_FILES).touch(exist_ok=True)
    with open(IMAGESET_TRAIN_FILES, "w") as f:
        for file in flatten(output):
            f.write(f"{file}\n")

    Path(OUTPUT_LABELS).touch(exist_ok=True)
    with open(OUTPUT_LABELS, "w") as f:
        for label in input.keys():
            f.write(f"{label}\n")

    background = glob.glob(load["test"])

    Path(IMAGESET_TEST_FILES).touch(exist_ok=True)
    with open(IMAGESET_TEST_FILES, "w") as f:
        for label, dir in ignore.items():
            files = [Path(x) for x in glob.glob(original[label])]
            for file in dir:
                filename, ext = os.path.splitext(os.path.basename(file))
                base = [x for x in files if x.stem.lower() == filename.lower()][0]
                data = ImageProcessor.from_path_base(file, str(base))
                copy = data.copy()
                source_x, source_y = copy.get_size()
                copy.resize_axis_x(OUTPUT_SIZE, True) if source_x > source_y else copy.resize_axis_y(OUTPUT_SIZE, True)
                copy.square(base=True)
                copy2 = copy.copy()
                copy2.set_base(cv2.cvtColor(NoiseImage().generate(N=OUTPUT_SIZE, count=5).astype("uint8"), cv2.COLOR_BGR2BGRA))
                copy.paste().write(f"{OUTPUT_IGNORE}/{filename}_base{OUTPUT_EXT}")
                copy2.paste().write(f"{OUTPUT_IGNORE}/{filename}_noise{OUTPUT_EXT}")

                back = ImageProcessor.from_path(random.choice(background))
                data = ImageProcessor.from_path(file)
                back_source_x, back_source_y = back.get_size()
                back.resize_axis_x(OUTPUT_SIZE) if back_source_x < back_source_y else back.resize_axis_y(OUTPUT_SIZE)
                x, y = back.get_size()
                back.trim(
                    (x - OUTPUT_SIZE) // 2,
                    (y - OUTPUT_SIZE) // 2,
                    (x + OUTPUT_SIZE) // 2,
                    (y + OUTPUT_SIZE) // 2,
                )
                randsize = random.randint(OUTPUT_SIZE // 4, (OUTPUT_SIZE - OUTPUT_SIZE // 4))
                data = random_resize(data.remove_noise(), OUTPUT_SIZE, randsize)
                data = data.set_base_value(back)

                annotate_file(data, label, filename)
                data.write(f"{OUTPUT_IGNORE}/{filename}{OUTPUT_EXT}")
                f.write(f"{filename}\n")
