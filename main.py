import glob
import pathlib
import shutil
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import MatLike
from rembg import remove
from tqdm import tqdm

from annotate import annotate


def print(*args, **kwargs):
    tqdm.write(" ".join(map(str, args)), **kwargs)


class ImageProcessor:
    value: MatLike
    """画像のデータを保持する"""
    base: MatLike
    """元画像のデータを保持する"""

    def __init__(self, input: MatLike, base: MatLike):
        """画像のデータと元画像のデータを保持する"""
        self.value = input
        self.base = base

    @staticmethod
    def from_path(input_path: str) -> "ImageProcessor":
        """ファイルのパスから画像を読み込む"""
        data = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if data.shape[2] == 3:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2BGRA)
        return ImageProcessor(data, data.copy())

    def copy(self) -> "ImageProcessor":
        """データをコピーする"""
        return ImageProcessor(self.value.copy(), self.base.copy())

    def set(self) -> "ImageProcessor":
        """元画像のデータを更新する"""
        self.base = self.value.copy()
        return self

    def marge(self, x: int, y: int, w: int, h: int, base: "ImageProcessor"):
        """現在の画像データと元画像データをマージする"""
        value = np.zeros_like(base.base)
        value[y:h, x:w] = self.value
        return ImageProcessor(value, base.base.copy())

    def trim(self, x: int, y: int, w: int, h: int) -> "ImageProcessor":
        """画像をトリミングする"""
        self.value = self.value[y:h, x:w]
        return self

    def resize(self, size: int) -> "ImageProcessor":
        """画像を余白付きでリサイズする"""
        value = np.zeros_like(self.value)
        value[size:-size, size:-size] = cv2.resize(
            self.value, (self.value.shape[1] - 2 * size, self.value.shape[0] - 2 * size)
        )
        self.value = value
        return self

    def add_border(self) -> "ImageProcessor":
        """画像に境界線を追加する"""
        x, y, w, h = cv2.boundingRect(cv2.cvtColor(self.value, cv2.COLOR_BGRA2GRAY))
        self.value = cv2.rectangle(self.value, (x, y), (x + w, y + h), (255, 0, 0, 255), 1)
        return self

    def add_contour(self) -> "ImageProcessor":
        """画像に輪郭線を追加する"""
        gray = cv2.cvtColor(self.value, cv2.COLOR_BGRA2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.value = cv2.drawContours(self.value, contours, -1, (0, 255, 0, 255), 1)
        return self

    def get_border_size(self) -> tuple[int, int, int, int]:
        """画像の境界線の座標を取得する"""
        x, y, w, h = cv2.boundingRect(cv2.cvtColor(self.value, cv2.COLOR_BGRA2GRAY))
        return x, y, x + w, y + h

    def get_size(self) -> tuple[int, int]:
        """画像のサイズを取得する"""
        return self.value.shape[1], self.value.shape[0]

    def get_mask_size(self) -> int:
        """画像のマスク部分の面積を取得する"""
        mask = self.value[:, :, 3] == 255
        return mask.sum()

    def paste(self) -> "ImageProcessor":
        """画像を貼り付ける"""
        mask = self.value[:, :, 3] == 0
        self.value[mask] = self.base[mask]
        return self

    def remove_background(self) -> "ImageProcessor":
        """画像の背景を透過する"""
        self.value = remove(self.value).copy()  # type: ignore
        mask = self.value[:, :, 3] > 150
        self.value = np.zeros_like(self.value)
        self.value[mask] = self.base[mask]
        return self

    def one_object(self) -> "ImageProcessor":
        """画像の一番大きいオブジェクト以外を削除する"""
        gray = cv2.cvtColor(self.value, cv2.COLOR_BGRA2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        other = sorted(contours, key=cv2.contourArea, reverse=True)[1:]
        self.value = cv2.drawContours(self.value, other, -1, (0, 0, 0, 0), -1)
        return self

    def write(self, output_path: str) -> "ImageProcessor":
        """画像を保存する"""
        cv2.imwrite(output_path, self.value)
        return self

    def show(self, block=True) -> "ImageProcessor":
        """画像を表示する"""
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(self.value, cv2.COLOR_BGRA2RGBA))
        plt.axis("off")
        plt.show(block=block)
        return self

    def compare(self, other: "ImageProcessor") -> float:
        data1 = cv2.calcHist([self.value], [0], None, [256], [0, 256])
        data2 = cv2.calcHist([other.value], [0], None, [256], [0, 256])
        value = cv2.compareHist(data1, data2, cv2.HISTCMP_CORREL)
        print(value)
        return value


def show_images_non_block(images: list[ImageProcessor]):
    for i, image in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(cv2.cvtColor(image.value, cv2.COLOR_BGRA2RGBA))
        plt.axis("off")
    plt.show(block=False)
    return plt.close


def change_base_dir(base: pathlib.Path, path: str, suffix: str) -> pathlib.Path:
    new = pathlib.Path(path) / base.relative_to("input").with_suffix(suffix)
    new.parent.mkdir(parents=True, exist_ok=True)
    return new


def get_area(x: int, y: int, w: int, h: int) -> int:
    return (w - x) * (h - y)


def get_resize(x: int, y: int, size: int) -> tuple[int, int, int, int]:
    return -size, -size, x + size, y + size


def get_resize2(x: int, y: int, w: int, h: int, pad: int) -> tuple[int, int, int, int]:
    return x - pad, y - pad, w + pad, h + pad


def any_compare(data: ImageProcessor, role: list[ImageProcessor]) -> bool:
    for x in role:
        if data.compare(x) > 0.9995:
            return True
    return False


def get_ichigo(base: ImageProcessor, role: list[ImageProcessor]) -> Union[ImageProcessor, list[ImageProcessor]]:
    data = base.copy().remove_background().one_object()
    border_size, size = (
        data.get_border_size(),
        data.get_size(),
    )

    trim1 = get_resize2(*border_size, -10)
    trim2 = get_resize(*size, -10)

    data1 = base.copy().trim(*trim1).set().remove_background().one_object()
    data2 = base.copy().trim(*trim2).set().remove_background().one_object()
    value = [data, data1.marge(*trim1, base), data2.marge(*trim2, base)]

    listed = filter(lambda x: x.get_mask_size() > 100 and any_compare(x, role), value)
    d = sorted(listed, key=lambda x: x.get_mask_size(), reverse=True)
    if len(d) == 0:
        return list(value)
    else:
        return d[0]


if __name__ == "__main__":
    shutil.rmtree("output", ignore_errors=True)
    shutil.rmtree("debug", ignore_errors=True)
    role_data: list[ImageProcessor] = [ImageProcessor.from_path(x) for x in glob.glob("role/*/*.png")]

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
