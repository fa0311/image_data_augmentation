import glob
import pathlib
import random
import shutil

from tqdm import tqdm

from annotate import annotate
from lib.processor import ImageProcessor
from lib.util import change_dir

# threshold = 0.9999996
threshold = 0.9999
remove = True
default_value = True


def print(*args, **kwargs):
    tqdm.write(" ".join(map(str, args)), **kwargs)


def ramdom_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255)


def data_augmentation(base: ImageProcessor) -> list[ImageProcessor]:
    res = []
    for i in range(4):
        data = base.copy()
        count = 0
        while random.random() < 0.5 or count < 1:
            count += 1
            mode = random.randint(0, 1)
            if mode == 0:
                data.rotate(random.randint(0, 360))
                data.flip(random.randint(-1, 1))
            # elif mode == 3:
            #     data.contrast(random.randint(0, 30))
            # elif mode == 4:
            #     data.blur(random.randint(0, 1) * 2 + 1)
            # elif mode == 5:
            #     data.sharpen()
            # elif mode == 6:
            #     data.noise(random.randint(1, 30))
            # elif mode == 7:
            #     data.hue(random.randint(-30, 30))
            # elif mode == 8:
            #     data.saturation(random.randint(-30, 30))
            # elif mode == 9:
            #     data.brightness(random.randint(-30, 30))

        res.append(data.copy())
    return res


if __name__ == "__main__":
    base = pathlib.Path("output2")
    if remove:
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(exist_ok=True, parents=True)

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
            for file in glob.glob("output1/image/*/*.png"):
                f.write(f"{pathlib.Path(file).stem}\n")

    for file in tqdm(glob.glob("output1/image/*/*.png")):
        base_path = pathlib.Path(file)
        output_path = change_dir(base_path, "output2/JPEGImages", ".jpg")
        output_annotation_path = change_dir(base_path, "output2/Annotations", ".xml")
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
            data.copy().paste().write(str(output_path.with_name(output_path.stem + f"_{i}.jpg")))
            annotation = annotate(output_path, border_size, size, label)
            annotation.write(str(output_annotation_path.with_name(output_annotation_path.stem + f"_{i}.xml")))
