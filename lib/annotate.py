import xml.etree.ElementTree as ET
from pathlib import Path


def dict_to_xml(pearent, d):
    for key, val in d.items():
        if isinstance(val, dict):
            elem = ET.Element(key)
            dict_to_xml(elem, val)
            pearent.append(elem)
        else:
            child = ET.SubElement(pearent, key)
            child.text = str(val)


def create_root_element(json_data):
    root = ET.Element("annotation", verified="yes")
    dict_to_xml(root, json_data)
    return root


def annotate(
    path: Path,
    bndbox: tuple[int, int, int, int],
    size: tuple[int, int],
    label: str,
) -> ET.ElementTree:
    data = {
        "folder": "Annotation",
        "filename": path.name,
        "path": path.as_posix(),
        "source": {"database": "Unknown"},
        "size": {"width": size[0], "height": size[1], "depth": 3},
        "segmented": 0,
        "object": {
            "name": label,
            "pose": "Unspecified",
            "truncated": 0,
            "difficult": 0,
            "bndbox": {"xmin": bndbox[0], "ymin": bndbox[1], "xmax": bndbox[2], "ymax": bndbox[3]},
        },
    }
    root = create_root_element(data)
    return ET.ElementTree(root)
