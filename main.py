import glob
import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import MatLike
from rembg import remove
from tqdm import tqdm

from annotate import annotate

print = tqdm.write


class DataAugmentation:
    def __init__(self, input: MatLike, base: MatLike):
        self.value: MatLike = input
        self.base: MatLike = base

    @staticmethod
    def from_path(input_path: str):
        data = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if data.shape[2] == 3:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2BGRA)
        return DataAugmentation(data, data.copy())

    def base_copy(self):
        return DataAugmentation(self.base.copy(), self.base.copy())

    def copy(self):
        return DataAugmentation(self.value.copy(), self.base.copy())

    def set(self):
        self.base = self.value.copy()
        return self

    def marge(self, x: int, y: int, w: int, h: int, base: "DataAugmentation"):
        value = np.zeros_like(base.base)
        value[y:h, x:w] = self.value
        return DataAugmentation(value, base.base)

    def trim(self, x: int, y: int, w: int, h: int):
        self.value = self.value[y:h, x:w]
        return self

    def add_border(self):
        x, y, w, h = cv2.boundingRect(cv2.cvtColor(self.value, cv2.COLOR_BGRA2GRAY))
        self.value = cv2.rectangle(self.value, (x, y), (x + w, y + h), (255, 0, 0, 255), 1)
        return self

    def add_contour(self):
        gray = cv2.cvtColor(self.value, cv2.COLOR_BGRA2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.value = cv2.drawContours(self.value, contours, -1, (0, 255, 0, 255), 1)
        return self

    def get_border_size(self):
        x, y, w, h = cv2.boundingRect(cv2.cvtColor(self.value, cv2.COLOR_BGRA2GRAY))
        return x, y, x + w, y + h

    def get_size(self):
        return self.value.shape[1], self.value.shape[0]

    def get_mask_size(self):
        mask = self.value[:, :, 3] == 255
        return mask.sum()

    def paste(self):
        mask = self.value[:, :, 3] == 0
        self.value[mask] = self.base[mask]
        return self

    def remove(self):
        self.value = remove(self.value).copy()  # type: ignore
        mask = self.value[:, :, 3] > 150
        self.value = np.zeros_like(self.value)
        self.value[mask] = self.base[mask]
        return self

    def one_object(self):
        gray = cv2.cvtColor(self.value, cv2.COLOR_BGRA2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        other = sorted(contours, key=cv2.contourArea, reverse=True)[1:]
        self.value = cv2.drawContours(self.value, other, -1, (0, 0, 0, 0), -1)
        return self

    def write(self, output_path: str):
        cv2.imwrite(output_path, self.value)
        return self

    def show(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(self.value, cv2.COLOR_BGRA2RGBA))
        plt.axis("off")
        plt.show()
        return self


def change_base_dir(base: pathlib.Path, path: str, suffix: str) -> pathlib.Path:
    new = pathlib.Path(path) / base.relative_to("input").with_suffix(suffix)
    new.parent.mkdir(parents=True, exist_ok=True)
    return new


def get_area(x: int, y: int, w: int, h: int) -> int:
    return (w - x) * (h - y)


def get_resize(x: int, y: int, w: int, h: int, size: int) -> tuple[int, int, int, int]:
    return x - size, y - size, w + size, h + size


if __name__ == "__main__":
    for dir in tqdm(glob.glob("input/*")):
        for file in tqdm(glob.glob(f"{dir}/*.jpg")):
            base = pathlib.Path(file)
            output_path = change_base_dir(base, "output/image", ".png")
            output_annotation_path = change_base_dir(base, "output/annotation", ".xml")
            debug_path = change_base_dir(base, "debug", ".png")

            data = DataAugmentation.from_path(file).remove().one_object()
            border_size, size, mask_size = (
                data.get_border_size(),
                data.get_size(),
                data.get_mask_size(),
            )

            trim = get_resize(*border_size, 10)
            data2 = data.base_copy().trim(*trim).set().remove()
            border_size2, size2, mask_size2 = (
                data2.get_border_size(),
                data2.get_size(),
                data2.get_mask_size(),
            )

            if get_area(*border_size2) / (size2[0] * size2[1]) > 0.8:
                pass
            else:
                print(f"{file} is too big, restore...")
                if __debug__:
                    retry = debug_path.with_name(debug_path.stem + "_retry" + debug_path.suffix)
                    data.copy().add_contour().add_border().paste().write(str(retry))

                data = data2.marge(*trim, data)
                border_size, size, mask_size = (
                    data.get_border_size(),
                    data.get_size(),
                    data.get_mask_size(),
                )

            data.copy().write(str(output_path))
            data.copy().add_contour().add_border().paste().write(str(debug_path))
            label = pathlib.Path(file).parent.name

            annotation = annotate(output_path, border_size, size, label)
            annotation.write(str(output_annotation_path))
