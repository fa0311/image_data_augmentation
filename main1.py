import glob
import pathlib
import shutil
from typing import Optional, Union

from tqdm import tqdm

from lib.processor import ImageProcessor
from lib.util import change_base_dir, get_resize, get_resize2, show_images_non_block

# threshold = 0.9999996
threshold = 0.9999
remove = True
default_value = True


def print(*args, **kwargs):
    tqdm.write(" ".join(map(str, args)), **kwargs)


def any_compare(data: ImageProcessor, role: list[ImageProcessor]) -> bool:
    for x in role:
        if data.compare(x) > threshold:
            return True
    return False


def get_ichigo(base: ImageProcessor, role: list[ImageProcessor]) -> Union[ImageProcessor, list[ImageProcessor]]:
    size = base.get_size()
    thickness = size[0] // 20
    trim = get_resize(*size, -thickness)

    data1 = base.copy().remove_background().one_object()
    data2 = base.copy().trim(*trim).set().remove_background().one_object().marge(*trim, base)

    border_size1 = data1.get_border_size()
    border_size2 = data2.get_border_size()

    trim11 = get_resize2(*border_size1, -thickness)
    trim12 = get_resize2(*border_size1, thickness)
    trim21 = get_resize2(*border_size2, -thickness)
    trim22 = get_resize2(*border_size2, thickness)

    data11 = base.copy().trim(*trim11).set().remove_background().one_object().marge(*trim11, base)
    data12 = base.copy().trim(*trim12).set().remove_background().one_object().marge(*trim12, base)
    data21 = base.copy().trim(*trim21).set().remove_background().one_object().marge(*trim21, base)
    data22 = base.copy().trim(*trim22).set().remove_background().one_object().marge(*trim22, base)

    all = [data1, data2, data11, data12, data21, data22]

    s = size[0] * size[1] // 30
    listed = filter(lambda x, s=s: x.get_mask_size() > s and any_compare(x, role), all)
    value = sorted(listed, key=lambda x: x.get_mask_size(), reverse=True)
    if len(role) == 0:
        return list(all)
    elif data1 == value[0] and data12 in value:
        return data12
    elif data2 == value[0] and data22 in value:
        return data22
    elif default_value:
        return data22
    elif len(value) < 2:
        return list(all)
    else:
        return value[0]


def ask(data: list[ImageProcessor]) -> Optional[ImageProcessor]:
    block = show_images_non_block([x.copy().add_contour().add_border().paste() for x in data])
    value = input("Select the correct image number: ")
    block()
    if value.isdigit() and int(value) < len(data):
        return data[int(value)]

    return None


if __name__ == "__main__":
    if remove:
        shutil.rmtree("output1", ignore_errors=True)
    role_data: list[ImageProcessor] = [ImageProcessor.from_path(x) for x in glob.glob("role/*/*.png")]  # noqa: F821

    for file in tqdm(glob.glob("input/*/*.JPG")):
        base_path = pathlib.Path(file)
        output_path = change_base_dir(base_path, "output1/image", ".png")
        debug_path = change_base_dir(base_path, "output1/debug", ".png")
        role_path = change_base_dir(base_path, "output1/role", ".png")
        error_path = change_base_dir(base_path, "output1/error", ".png")

        if output_path.exists():
            continue

        base = ImageProcessor.from_path(str(base_path))
        print(base_path)
        data_list = get_ichigo(base, role_data)
        if isinstance(data_list, list):
            data = ask(data_list)
            if data is None:
                base.copy().write(str(error_path))
            else:
                data.copy().write(str(role_path))
                role_data.append(data)
        else:
            data = data_list
        if data is not None:
            border_size, size = data.get_border_size(), data.get_size()
            data.copy().write(str(output_path))
            data.copy().add_contour().add_border().paste().write(str(debug_path))
            # label = pathlib.Path(file).parent.name
            # annotation = annotate(output_path, border_size, size, label)
            # annotation.write(str(output_annotation_path))
