import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import MatLike
from rembg import remove
from tqdm import tqdm


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
        thickness = self.value.shape[0] // 500
        self.value = cv2.rectangle(self.value, (x, y), (x + w, y + h), (255, 0, 0, 255), thickness)
        return self

    def add_contour(self) -> "ImageProcessor":
        """画像に輪郭線を追加する"""
        gray = cv2.cvtColor(self.value, cv2.COLOR_BGRA2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thickness = self.value.shape[0] // 500
        self.value = cv2.drawContours(self.value, contours, -1, (0, 255, 0, 255), thickness)
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
