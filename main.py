import glob
import json
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
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
IMAGESET_TRAIN_FILES = f"{OUTPUT_IMAGESET}/trainval.txt"
IMAGESET_TEST_FILES = f"{OUTPUT_IMAGESET}/test.txt"


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
        annotate_file(data, label, f"{filename}_{i}")


def annotate_file(data: ImageProcessor, label: str, filename: str):
    border_size, _ = data.get_border_size(), data.get_size()
    data.paste().write(f"{OUTPUT_JPEG}/{filename}{OUTPUT_EXT}")
    path_anno = f"{OUTPUT_ANNOTATION}/{filename}{OUTPUT_EXT}"
    annotated = annotate(Path(path_anno), border_size, (OUTPUT_SIZE, OUTPUT_SIZE), label)
    annotated.write(f"{OUTPUT_ANNOTATION}/{filename}.xml")


if __name__ == "__main__":
    with open("input.json") as f:
        load: dict[str, dict[str, str]] = json.load(f)
    input = {k: v["input"] for k, v in load.items()}
    original = {k: v["original"] for k, v in load.items()}

    ignore = {label: random.sample(glob.glob(path), 5) for label, path in input.items()}

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_JPEG, exist_ok=True)
    os.makedirs(OUTPUT_ANNOTATION, exist_ok=True)
    os.makedirs(OUTPUT_IMAGESET, exist_ok=True)
    os.makedirs(OUTPUT_IGNORE, exist_ok=True)

    file_list = [[(x, label) for x in glob.glob(path) if x not in ignore[label]] for label, path in input.items()]
    files = flatten(file_list)
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_file, files), total=len(files), desc="Files", leave=False))

    Path(IMAGESET_TRAIN_FILES).touch(exist_ok=True)
    with open(IMAGESET_TRAIN_FILES, "w") as f:
        for file in files:
            f.write(f"{Path(file[0]).stem}\n")

    Path(OUTPUT_LABELS).touch(exist_ok=True)
    with open(OUTPUT_LABELS, "w") as f:
        for label in input.keys():
            f.write(f"{label}\n")

    Path(IMAGESET_TEST_FILES).touch(exist_ok=True)
    with open(IMAGESET_TEST_FILES, "w") as f:
        for label, dir in ignore.items():
            files = [Path(x) for x in glob.glob(original[label])]
            for file in dir:
                filename, ext = os.path.splitext(os.path.basename(file))
                base = [x for x in files if x.stem.lower() == filename.lower()][0]
                data = ImageProcessor.from_path_base(file, str(base))
                source_x, source_y = data.get_size()
                data.resize_axis_x(OUTPUT_SIZE, True) if source_x > source_y else data.resize_axis_y(OUTPUT_SIZE, True)
                annotate_file(data, label, filename)
                data.write(f"{OUTPUT_IGNORE}/{filename}{OUTPUT_EXT}")
                f.write(f"{filename}\n")
