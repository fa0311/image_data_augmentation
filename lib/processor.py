import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import MatLike


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

    @staticmethod
    def from_path_without_base(input_path: str) -> "ImageProcessor":
        """ファイルのパスから画像を読み込む"""
        data = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if data.shape[2] == 3:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2BGRA)
        return ImageProcessor(data, np.zeros_like(data))

    @staticmethod
    def from_path_base(input_path: str, base_path: str) -> "ImageProcessor":
        """ファイルのパスから画像を読み込む"""
        data = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if data.shape[2] == 3:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2BGRA)
        base = cv2.imread(base_path, cv2.IMREAD_UNCHANGED)
        if base.shape[2] == 3:
            base = cv2.cvtColor(base, cv2.COLOR_BGR2BGRA)
        return ImageProcessor(data, base)

    @staticmethod
    def from_without_base(size: tuple[int, int]) -> "ImageProcessor":
        return ImageProcessor(np.zeros((*size, 4), np.uint8), np.zeros((*size, 4), np.uint8))

    def copy(self) -> "ImageProcessor":
        """データをコピーする"""
        return ImageProcessor(self.value.copy(), self.base.copy())

    def set(self) -> "ImageProcessor":
        """元画像のデータを更新する"""
        self.base = self.value.copy()
        return self

    def set_base(self, base: MatLike) -> "ImageProcessor":
        self.base = base
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

    def add_margin(self, margin: tuple[int, int, int, int]) -> "ImageProcessor":
        """画像に余白を追加する"""
        top, right, bottom, left = margin
        value = np.zeros((self.value.shape[0] + top + bottom, self.value.shape[1] + left + right, 4), np.uint8)
        value[top:-bottom, left:-right] = self.value
        self.value = value
        return self

    def square(self, base: bool = False) -> "ImageProcessor":
        size = max(self.value.shape[0], self.value.shape[1])
        value = np.zeros((size, size, 4), np.uint8)
        value[: self.value.shape[0], : self.value.shape[1]] = self.value
        self.value = value
        if base:
            value = np.zeros((size, size, 4), np.uint8)
            value[: self.base.shape[0], : self.base.shape[1]] = self.base
            self.base = value
        return self

    def resize_axis_x(self, size: int, base: bool = False) -> "ImageProcessor":
        """画像をアスペクト比を保持してリサイズする"""
        ratio = size / self.value.shape[1]
        self.value = cv2.resize(self.value, (size, int(self.value.shape[0] * ratio)))
        if base:
            self.base = cv2.resize(self.base, (size, int(self.base.shape[0] * ratio)))
        return self

    def resize_axis_y(self, size: int, base: bool = False) -> "ImageProcessor":
        """画像をアスペクト比を保持してリサイズする"""
        ratio = size / self.value.shape[0]
        self.value = cv2.resize(self.value, (int(self.value.shape[1] * ratio), size))
        if base:
            self.base = cv2.resize(self.base, (int(self.base.shape[1] * ratio), size))
        return self

    def add_border(self) -> "ImageProcessor":
        """画像に境界線を追加する"""
        x, y, w, h = cv2.boundingRect(cv2.cvtColor(self.value, cv2.COLOR_BGRA2GRAY))
        thickness = self.value.shape[0] // 500
        self.value = cv2.rectangle(self.value, (x, y), (x + w, y + h), (255, 0, 0, 255), thickness)
        return self

    def set_base_color(self, color: tuple[int, int, int, int]) -> "ImageProcessor":
        """画像の背景色を変更する"""
        self.base = np.zeros_like(self.value)
        self.base[:, :] = color
        return self

    def rotate(self, angle: float) -> "ImageProcessor":
        """画像を回転する"""
        center = (self.value.shape[1] // 2, self.value.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        self.value = cv2.warpAffine(self.value, matrix, (self.value.shape[1], self.value.shape[0]))
        return self

    def flip(self, flip_code: int) -> "ImageProcessor":
        """画像を反転する"""
        self.value = cv2.flip(self.value, flip_code)
        return self

    def hsv(self, hue: int, saturation: float, value: float) -> "ImageProcessor":
        """画像の色相、彩度、明度を変更する"""
        alpha = self.value[:, :, 3]
        hsv = cv2.cvtColor(self.value, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * value, 0, 255)
        hsv = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        self.value = cv2.cvtColor(hsv, cv2.COLOR_BGR2BGRA)
        self.value[:, :, 3] = alpha
        return self

    # 輪郭の付近をぼかす
    def blur_contour(self) -> "ImageProcessor":
        """画像の輪郭の付近をぼかす"""
        alpha_channel = self.value[:, :, 3]
        _, mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)
        dilated_mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        blurred_image = cv2.GaussianBlur(self.value[:, :, :3], (5, 5), 0)
        self.value[:, :, :3] = np.where(dilated_mask[:, :, None] == 255, blurred_image, self.value[:, :, :3])
        return self

    def remove_noise(self) -> "ImageProcessor":
        alpha = self.value[:, :, 3]
        edges = cv2.Canny(alpha, 100, 200)
        dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        alpha[(dilated_edges > 0)] = 0
        self.value[:, :, 3] = alpha
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

    # def remove_background(self) -> "ImageProcessor":
    #     """画像の背景を透過する"""
    #     self.value = remove(self.value).copy()  # type: ignore
    #     mask = self.value[:, :, 3] > 150
    #     self.value = np.zeros_like(self.value)
    #     self.value[mask] = self.base[mask]
    #     return self

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
        data1 = cv2.calcHist([self.value], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        data2 = cv2.calcHist([other.value], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        value = cv2.compareHist(data1, data2, cv2.HISTCMP_CORREL)
        return value
