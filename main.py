import glob
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import MatLike


class ImageProcessing:
    def __init__(self, image: MatLike):
        self.image = image
        self.delete = []
        self.keep = []

    def gray(self, threshold=128):
        return np.max(self.image, axis=2) - np.min(self.image, axis=2) < threshold

    def thin(self, threshold=128):
        return np.mean(self.image, axis=2) > threshold  # type: ignore

    def color(self, threshold=128, color=0):
        """その色のみ他の色よりも大きいかどうか"""
        mask1 = self.image[:, :, color] - self.image[:, :, (color + 1) % 3] > threshold
        mask2 = self.image[:, :, color] - self.image[:, :, (color + 2) % 3] > threshold
        return mask1 & mask2

    def color2(self, threshold=128, color=0):
        """その色が大きいかどうか"""
        return self.image[:, :, color] > threshold

    def color3(self, threshold=128, color=0):
        """その色が最大値よりもいくつか大きいかどうか"""
        return np.max(self.image, axis=2) - self.image[:, :, color] > threshold

    def blue(self, threshold=128):
        return self.color(threshold, 0)

    def green(self, threshold=128):
        return self.color(threshold, 1)

    def red(self, threshold=128):
        return self.color(threshold, 2)

    def blue2(self, threshold=128):
        return self.color2(threshold, 0)

    def green2(self, threshold=128):
        return self.color2(threshold, 1)

    def red2(self, threshold=128):
        return self.color2(threshold, 2)

    def blue3(self, threshold=128):
        return self.color3(threshold, 0)

    def green3(self, threshold=128):
        return self.color3(threshold, 1)

    def red3(self, threshold=128):
        return self.color3(threshold, 2)


class EdgeDetection:
    def __init__(self, image: MatLike):
        self.image = image
        self.base = image.copy()

    @staticmethod
    def from_file(png_path):
        image = cv2.imread(png_path, cv2.IMREAD_COLOR)
        assert image is not None
        return EdgeDetection(image)

    def set_base(self):
        self.base = self.image.copy()
        return self

    def trim(self, threshold=255, x=0, y=0):
        """画像をトリミングする"""
        mask = np.all(self.image > threshold, axis=2)
        mask = np.logical_not(mask)
        self.image = self.image[np.ix_(mask.any(1), mask.any(0))]
        self.image = self.image[y:-y, x:-x]
        return self

    def padding(self, x=0, y=0, color=(255, 255, 255)):
        """周りに余白を追加する"""
        h, w = self.image.shape[:2]
        self.image = cv2.copyMakeBorder(self.image, y, y, x, x, cv2.BORDER_CONSTANT, value=color)
        return self

    def copy(self):
        return EdgeDetection(self.image.copy())

    def gray(self):
        """画像をグレースケールに変換する"""
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return self

    def color(self, keep: Callable[[ImageProcessing], np.ndarray]):
        mask = keep(ImageProcessing(self.image))
        self.image = np.where(mask[:, :, None], self.image, 255)
        return self

    def enhance(self, color: int, n: float):
        """画像の色を強調する"""
        self.image[:, :, color] = np.clip(self.image[:, :, color] * n, 0, 255)
        return self

    def otsu_threshold(self):
        _, self.image = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self

    def gaussian_blur(self, kernel_size=5):
        self.image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)
        return self

    def sobel(self):
        sobelx = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=5)
        sobel_edges = cv2.magnitude(sobelx, sobely)
        self.image = cv2.convertScaleAbs(sobel_edges)
        return self

    def laplacian(self):
        laplacian = cv2.Laplacian(self.image, cv2.CV_64F)
        self.image = cv2.convertScaleAbs(laplacian)
        return self

    def morphological_gradient(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(self.image, cv2.MORPH_GRADIENT, kernel)
        self.image = cv2.convertScaleAbs(gradient)
        return self

    def canny(self, threshold1=50, threshold2=250):
        self.image = cv2.Canny(self.image, threshold1, threshold2)
        return self

    def find_and_remove_contours(self):
        contours, _ = cv2.findContours(self.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        mask = np.ones(self.image.shape, dtype=np.uint8) * 255
        mask = cv2.drawContours(mask, [large[0]], -1, (0, 0, 0), thickness=cv2.FILLED)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        kernel = np.ones((4, 4), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.bitwise_not(mask)
        self.image = cv2.bitwise_and(self.base, self.base, mask=mask)
        return self

    def kernel(self, kernel_size=2):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.image = cv2.dilate(self.image, kernel, iterations=1)
        return self

    def show(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
        return self


if __name__ == "__main__":
    for file_name in glob.glob("input/*"):
        image = EdgeDetection.from_file(file_name)

        # image.copy().enhance(0, 2).show()

        # image.copy().trim(200).set_base().enhance(0, 1.5).canny().find_contours().show()

        # image.copy().canny().find_contours().show()
        # image.copy().enhance(0, 1.5).show()

        # image.copy().color(lambda x: ~(x.thin(100))).show()

        # image.copy().color(lambda x: (~(x.gray(55)))).show()

        image.trim(200, x=80, y=80).set_base()
        image.copy().canny(50, 200).kernel(2).find_and_remove_contours().show()

        # image.copy().canny().find_contours().show()

        # image.copy().gray().canny().find_contours().show()

        # image.copy().gray().sobel().find_contours().show()

        # image.copy().gray().laplacian().find_contours().show()

        # image.copy().gray().morphological_gradient().find_contours().show()

        # image.copy().gray().gaussian_blur().canny().find_contours().show()

        # image.copy().gray().otsu_threshold().canny().find_contours().show()
