import os
import shutil

import cv2
from matplotlib import pyplot as plt

from lib.noise import NoiseImage

OUTPUT_DIR = "output"

if __name__ == "__main__":
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    noise = NoiseImage()

    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axs.flat):
        img = noise.generate(N=512, count=5)
        ax.imshow(cv2.cvtColor(img.astype("uint8"), cv2.COLOR_BGR2RGB))
        ax.axis("off")
        ax.set_title(f"{i}")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/perlin.png")
