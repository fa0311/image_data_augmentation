import glob
import pathlib
import shutil

from tqdm import tqdm

from annotate import annotate
from lib.processor import ImageProcessor
from lib.util import change_base_dir

if __name__ == "__main__":
    shutil.rmtree("output", ignore_errors=True)
    shutil.rmtree("debug", ignore_errors=True)

    for file in tqdm(glob.glob("input/*/*.JPG")):
        base_path = pathlib.Path(file)
        output_path = change_base_dir(base_path, "output/image", ".png")
        output_annotation_path = change_base_dir(base_path, "output/annotation", ".xml")
        debug_path = change_base_dir(base_path, "debug", ".png")
        role_path = change_base_dir(base_path, "role", ".png")

        data = ImageProcessor.from_path(str(base_path))
        data = data.remove_background().one_object()

        border_size, size = data.get_border_size(), data.get_size()
        data.copy().write(str(output_path))
        data.copy().add_contour().add_border().paste().write(str(debug_path))
        label = pathlib.Path(file).parent.name
        annotation = annotate(output_path, border_size, size, label)
        annotation.write(str(output_annotation_path))
