import xml.etree.ElementTree as ET
from pathlib import Path


def dict_to_xml(tag, d):
    elem = ET.Element(tag)
    for key, val in d.items():
        if isinstance(val, dict):
            child = dict_to_xml(key, val)
            elem.append(child)
        else:
            child = ET.SubElement(elem, key)
            child.text = str(val)
    return elem


def create_root_element(json_data):
    root_tag = list(json_data.keys())[0]
    root = ET.Element("annotation", verified="yes")
    root_element = dict_to_xml(root_tag, json_data)
    root.append(root_element)
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
