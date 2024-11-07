import glob
import json
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from math import ceil
from pathlib import Path

from tqdm import tqdm

from lib.annotate import annotate
from lib.processor import ImageProcessor

OUTPUT_DIR = "output"
OUTPUT_EXT = ".jpg"
OUTPUT_SIZE = 512

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
        res.append(data)
    return res


def random_resize(base: ImageProcessor, size: int) -> ImageProcessor:
    base.trim(*base.get_border_size())

    source_x, source_y = base.get_size()
    rand_size = random.randint(200, size - 100)
    base.resize_axis_x(rand_size) if source_x > source_y else base.resize_axis_y(rand_size)

    new_x, new_y = base.get_size()
    top_margin, right_margin = random.randint(1, size - new_y - 1), random.randint(1, size - new_x - 1)
    res = base.add_margin((top_margin, right_margin, size - new_y - top_margin, size - new_x - right_margin))
    return res


def ramdom_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)


def flatten(data):
    return [item for sublist in data for item in sublist]


def process_file(args):
    file, label = args
    filename, ext = os.path.splitext(os.path.basename(file))
    base_data = ImageProcessor.from_path_without_base(file)
    augmentation = data_augmentation(base_data)
    for i, data in enumerate(augmentation):
        random_resize(data.remove_noise(), OUTPUT_SIZE).set_base_color(ramdom_color())
        border_size, _ = data.get_border_size(), data.get_size()
        data.paste().write(f"{OUTPUT_JPEG}/{filename}_{i}{OUTPUT_EXT}")
        path_anno = f"{OUTPUT_ANNOTATION}/{filename}_{i}{OUTPUT_EXT}"
        annotated = annotate(Path(path_anno), border_size, (OUTPUT_SIZE, OUTPUT_SIZE), label)
        annotated.write(f"{OUTPUT_ANNOTATION}/{filename}_{i}.xml")


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

    file_list = [[(x, label) for x in glob.glob(path) if x not in ignore_flatten] for label, path in conf.items()]
    files = flatten(file_list)
    max_workers = ceil((os.cpu_count() or 1) / 4)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_file, files), total=len(files), desc="Files", leave=False))

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
