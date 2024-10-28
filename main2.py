import glob
import pathlib
import random
import shutil

from tqdm import tqdm

from lib.annotate import annotate
from lib.processor import ImageProcessor
from lib.util import change_base_dir, change_dir

remove = True


def print(*args, **kwargs):
    tqdm.write(" ".join(map(str, args)), **kwargs)


def flatten(data):
    return [item for sublist in data for item in sublist]


def ramdom_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)


def data_augmentation(base: ImageProcessor) -> list[ImageProcessor]:
    res = []
    for i in range(4):
        data = base.copy()
        data.rotate(random.randint(0, 360))
        data.flip(random.randint(-1, 1))

        res.append(data.copy())
    return res


if __name__ == "__main__":
    base = pathlib.Path("output2")
    if remove:
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(exist_ok=True, parents=True)
    ignore = [random.sample(glob.glob(f"{path}/*"), 5) for path in glob.glob("output1/image/*")]

    for file in tqdm(glob.glob("output1/image/*/*.png")):
        if file not in flatten(ignore):
            base_path = pathlib.Path(file)
            output_path = change_dir(base_path, "output2/JPEGImages", ".jpg")
            annotation_path = change_dir(base_path, "output2/Annotations", ".xml")
            label = pathlib.Path(file).parent.name
            data = ImageProcessor.from_path(str(base_path))

            # border_size, size = data.get_border_size(), data.get_size()
            # data.copy().paste().write(str(output_path))
            # annotation = annotate(output_path, border_size, size, label)
            # annotation.write(str(output_annotation_path))

            augmentation = data_augmentation(data)
            for i, data in enumerate(augmentation):
                data.resize_axis_x(512).set_base_color(ramdom_color())
                border_size, size = data.get_border_size(), data.get_size()
                annotation_path_augmentation = annotation_path.with_name(annotation_path.stem + f"_{i}.xml")
                annotation_path_augmentation_jpg = annotation_path.with_name(annotation_path.stem + f"_{i}.jpg")
                output_path_augmentation = output_path.with_name(output_path.stem + f"_{i}.jpg")

                annotation = annotate(annotation_path_augmentation_jpg, border_size, size, label)
                annotation.write(str(annotation_path_augmentation))
                data.copy().paste().write(str(output_path_augmentation))

    for dir in ignore:
        for file in dir:
            path = pathlib.Path(file)
            from_path = change_base_dir(path, "output1/image", "input", ".JPG")
            to_path = change_base_dir(path, "output1/image", "output2/ignore", ".jpg")
            data = ImageProcessor.from_path(str(from_path))
            data.copy().resize_axis_x(512).write(str(to_path))

    labels_path = base.joinpath("labels.txt")
    labels_path.touch(exist_ok=True)
    with open(labels_path, "a") as f:
        for file in glob.glob("output1/image/*"):
            f.write(pathlib.Path(file).name + "\n")

    image_set_path = base.joinpath("ImageSets/Main")
    image_set_path.mkdir(exist_ok=True, parents=True)
    for path_name in ["test.txt", "train.txt", "trainval.txt", "val.txt"]:
        path = image_set_path.joinpath(path_name)
        path.touch(exist_ok=True)
        with open(path, "w") as f:
            for file in glob.glob("output2/JPEGImages/*.jpg"):
                f.write(f"{pathlib.Path(file).stem}\n")
