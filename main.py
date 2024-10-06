import glob
import pathlib
import shutil
from typing import Union

from tqdm import tqdm

from annotate import annotate
from lib.processor import ImageProcessor
from lib.util import change_base_dir, get_resize, get_resize2, show_images_non_block


def any_compare(data: ImageProcessor, role: list[ImageProcessor]) -> bool:
    for x in role:
        if data.compare(x) > 0.9998:
            return True
    return False


def get_ichigo(base: ImageProcessor, role: list[ImageProcessor]) -> Union[ImageProcessor, list[ImageProcessor]]:
    data = base.copy().remove_background().one_object()
    border_size, size = (
        data.get_border_size(),
        data.get_size(),
    )

    thickness = size[0] // 100

    trim1 = get_resize2(*border_size, -thickness)
    trim2 = get_resize(*size, -thickness)

    data1 = base.copy().trim(*trim1).set().remove_background().one_object()
    data2 = base.copy().trim(*trim2).set().remove_background().one_object()
    value = [data, data1.marge(*trim1, base), data2.marge(*trim2, base)]
    value = [data]

    s = size[0] * size[1] // 100
    listed = filter(lambda x, s=s: x.get_mask_size() > s and any_compare(x, role), value)
    d = sorted(listed, key=lambda x: x.get_mask_size(), reverse=True)
    if len(d) == 0:
        return list(value)
    else:
        return d[0]


if __name__ == "__main__":
    shutil.rmtree("output", ignore_errors=True)
    shutil.rmtree("debug", ignore_errors=True)
    role_data: list[ImageProcessor] = [ImageProcessor.from_path(x) for x in glob.glob("role/*/*.png")]  # noqa: F821

    for file in tqdm(glob.glob("input/*/*.JPG")):
        base_path = pathlib.Path(file)
        output_path = change_base_dir(base_path, "output/image", ".png")
        output_annotation_path = change_base_dir(base_path, "output/annotation", ".xml")
        debug_path = change_base_dir(base_path, "debug", ".png")
        role_path = change_base_dir(base_path, "role", ".png")

        base = ImageProcessor.from_path(str(base_path))
        data = get_ichigo(base, role_data)
        if isinstance(data, list):
            block = show_images_non_block([x.copy().add_contour().add_border().paste() for x in data])
            value = input("Select the correct image number: ")
            block()
            if value.isdigit() and int(value) < len(data):
                data[int(value)].write(str(role_path))
                role_data.append(data[int(value)])
        else:
            border_size, size = data.get_border_size(), data.get_size()
            data.copy().write(str(output_path))
            data.copy().add_contour().add_border().paste().write(str(debug_path))
            label = pathlib.Path(file).parent.name
            annotation = annotate(output_path, border_size, size, label)
            annotation.write(str(output_annotation_path))
