import os
import numpy as np


def get_object(root):
    """ Getting object from XML annotation file 
    """
    return root.findall("./object")


def get_name(object):
    """ Getting name from the object in a XML annotation file
    """
    for name in object.findall("./name"):
        return name.text


def replace_name(object, old_name, new_name):
    """ Replacing name of old_name to new_name
    """
    for name in object.findall("./name"):
        if name.text == old_name:
            name.text = new_name


def get_bbox(object):
    """Get bounding box from the object

    Args:
        object ([type]): [description]

    Returns:
        [type]: [description]
    """
    for item in object.findall("./bndbox"):
        xmin, ymin, xmax, ymax = item[0].text, item[1].text, item[2].text, item[3].text
    return xmin, ymin, xmax, ymax


def get_image_filename(root):
    """Get image filename from root of XML annnotation

    Args:
        root ([type]): [description]

    Returns:
        [type]: [description]
    """
    return root.find("filename").text


def get_conf(object):
    """Get confidence level of object

    Args:
        object ([type]): [description]

    Returns:
        [type]: [description]
    """
    for item in object.findall("./bndbox"):
        conf = item[4].text
    return conf


def find_width_height_depth(root):
    size = root.find("./size")
    width, height, depth = int(size[0].text), int(size[1].text), int(size[2].text)
    return width, height, depth


## Changing XML annotations
def rescale(i, i_scale):
    return int(np.round(i * i_scale))


def shift(i, i_shift):
    return int(np.round(i + i_shift))


def change_bbox_coordinate(coordinate, scale_val, shift_val):
    coordinate = shift(rescale(coordinate, scale_val), shift_val)
    return coordinate


def replace_bbox(object, new_bbox):
    new_bbox = tuple(map(str, new_bbox))
    new_xmin, new_ymin, new_xmax, new_ymax = new_bbox
    for item in object.findall("./bndbox"):
        item[0].text = new_xmin
        item[1].text = new_ymin
        item[2].text = new_xmax
        item[3].text = new_ymax

    return


def find_dim(root):
    size = root.find("./size")
    old_x, old_y = int(size[0].text), int(size[1].text)
    return old_x, old_y


def replace_dim(root, new_dim):
    size = root.find("size")
    # convert all items in tuple to strings as xml require string format
    new_x, new_y = [str(i) for i in new_dim]
    size.find("width").text = new_x
    size.find("height").text = new_y
    return


def get_new_bbox_list(object, old_dim, new_dim):
    scale = new_dim[0] / max(old_dim)
    old_x, old_y = old_dim
    new_x, new_y = new_dim
    x_shift = (new_x - (old_x * scale)) / 2
    y_shift = (new_y - (old_y * scale)) / 2
    bbox_list = []
    for item in object.findall("./bndbox"):
        xmin, ymin, xmax, ymax = (
            int(item[0].text),
            int(item[1].text),
            int(item[2].text),
            int(item[3].text),
        )
        xmin = change_bbox_coordinate(xmin, scale, x_shift)
        ymin = change_bbox_coordinate(ymin, scale, y_shift)
        xmax = change_bbox_coordinate(xmax, scale, x_shift)
        ymax = change_bbox_coordinate(ymax, scale, y_shift)
        bbox = xmin, ymin, xmax, ymax
        bbox_list.append(bbox)
    return bbox_list


def remove_extension(root):
    """ Remove extension in filename
    """
    filename = root.find("filename").text
    filename_without_ext = filename.split(".")[0]
    root.find("filename").text = filename_without_ext
    return
