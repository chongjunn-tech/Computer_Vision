from ast import Import
import os
from pathlib import Path
import matplotlib.pyplot as plt
from random import sample
import xml.etree.ElementTree as ElementTree
import numpy as np
import shutil
from PIL import Image
import cv2
import re

import csv
import pandas as pd
from pascal_voc_writer import Writer
from .xml_change import *

amended_coco_labels = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "luggage",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def visualise_img_name_conf(image, names, confs, folder_img, bboxes, dpi=80):
    """Visualise image name and the confidence level (for model-generated annotations)

    Args:
        image ([type]): [description]
        names ([type]): [description]
        confs ([type]): [description]
        folder_img ([type]): [description]
        bboxes ([type]): [description]
        dpi (int, optional): [description]. Defaults to 80.
    """
    ## standardize all images to have file extension to be .jpg previously and remove all file extension from xml annot earlier
    image_file = f"{image}.jpg"
    image_array = cv2.imread(os.path.join(folder_img, image_file))

    color = (0, 0, 255)
    thickness = 2
    # get matplotlib object to return plot for all in the for loop
    _, ax = plt.subplots(dpi=dpi)
    for bbox, name, conf in zip(bboxes, names, confs):
        bbox = list(map(int, bbox))
        start = (bbox[0], bbox[1])

        end = (bbox[2], bbox[3])

        text = f"{name}: {conf}"
        cv2.rectangle(image_array, start, end, color, thickness)
        cv2.putText(
            image_array,
            text,
            (bbox[0] + 8, bbox[1] + 22),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            color,
        )

    ax.imshow(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    ax.set_title(image_file)

    return


def folder_annotation_all_bboxes_in_filename_dict(folder_annot):
    """Return in this format {image name:{COCO class:[[confidence,bbox in np array]]}}

    Args:
        folder_annot (str): path to annotation folder

    Returns:
        dict: {image name:{COCO class:[[confidence,bbox in np array]]}}
    """
    filename_dict = {}
    bboxes_count = 0
    for dirpath, dirnames, _ in os.walk(folder_annot):
        if not dirnames:
            for file in os.listdir(dirpath):
                if file.endswith(".xml"):
                    xml_annotation = os.path.join(dirpath, file)
                    tree = ElementTree.parse(xml_annotation)
                    root = tree.getroot()
                    image = get_image_filename(root)
                    for object in get_object(root):
                        name = get_name(object)
                        bbox = get_bbox(object)
                        conf = get_conf(object)
                        bboxes_count += 1
                        if image not in filename_dict:
                            filename_dict[image] = []
                        tmp_list = [
                            name,
                            float(conf),
                            np.array(tuple(map(float, bbox))),
                        ]
                        filename_dict[image].append(tmp_list)
    return filename_dict


def folder_annotation_all_bboxes_in_dict(folder_annot):
    """Return in this format {COCO CLASS:{image name:[[confidence,bbox in np array]]}}

    Args:
        folder_annot (str): path to annotation folder

    Returns:
        dict: {COCO CLASS:{image name:[[confidence,bbox in np array]]}}
    """
    xml_dict = {x: {} for x in amended_coco_labels}
    bboxes_count = 0

    for file in os.listdir(folder_annot):
        if file.endswith(".xml"):
            xml_annotation = os.path.join(folder_annot, file)
            tree = ElementTree.parse(xml_annotation)
            root = tree.getroot()
            image = get_image_filename(root)
            for object in get_object(root):
                name = get_name(object)
                bbox = get_bbox(object)
                conf = get_conf(object)
                bboxes_count += 1
                if image not in xml_dict[name]:
                    xml_dict[name][image] = []

                tmp_list = [float(conf), np.array(tuple(map(float, bbox)))]
                xml_dict[name][image].append(tmp_list)

    return xml_dict, bboxes_count


def visualise_folder_image_annotation(
    folder_img,
    folder_annot,
    all_files=True,
    sample_size=False,
    visualised_list=None,
    start=0,
    end=10,
    windows_sorted=False,
    dpi=80,
    visualise_confidence=False,
):
    """Visualise images in folder_img with its annotations in folder_annot

    Args:
        folder_img (str): path to folder containing image(s)
        folder_annot (str): path to folder containing annotation(s)
        all_files (bool, optional): visualise all files in folder_img. Defaults to True.
        sample_size (int): sample for visualisation. Defaults to False.
        visualised_list (list, optional): visualised images in visualised_list. Defaults to None.
        start (int, optional): starting index of visualised image. Defaults to 0.
        end (int, optional): End index of visualised image Defaults to 10.
        windows_sorted (bool, optional):if True, will sort the images name in windows explorer sorting format. Defaults to False.
        dpi (int, optional): dot per inches of visualisation. Defaults to 80.
        visualise_confidence (bool, optional): include the confidence of the annotation(if information is present in XML annotation). Defaults to False.

    Raises:
        Exception: if user input all_files and sample_size together. To use sample_size, all_files need to be set as
    """
    if all_files and sample_size:
        raise Exception("Cannot input visualised all_files and sample size")

    if visualised_list:
        visualised_list_without_ext = [f.split(".")[0] for f in visualised_list]
        xml_list = [
            f
            for f in os.listdir(folder_annot)
            if f.endswith("xml") and f.split(".")[0] in visualised_list_without_ext
        ]
    else:
        xml_list = [f for f in os.listdir(folder_annot) if f.endswith("xml")]

    # sorting order
    ## if user input it to be windows explorer sorted. For example,first 3 items will be n03709823_27.xml<'n03709823_84.xml'<'n04026417_95.xml'
    if windows_sorted:
        files_annot = sorted(xml_list, key=lambda x: split_file(x))
    ## else, will be python sorted. For example, first 3 items will be n03709823_10126.xml<'n03709823_10199.xml'<'n03709823_10210.xml'
    else:
        files_annot = xml_list

    # selection of files to visualise
    ## if select sample size
    if sample_size:
        files_to_visualised = sample(files_annot, sample_size)

    elif all_files:
        files_to_visualised = files_annot[start:end]

    for xml_annotation in files_to_visualised:
        if xml_annotation.endswith(".xml"):
            file = os.path.join(folder_annot, xml_annotation)
            tree = ElementTree.parse(file)
            root = tree.getroot()
            image = get_image_filename(root)

            bboxes = []
            names = []
            confs = []
            for object in get_object(root):
                name = get_name(object)
                bbox = get_bbox(object)

                bboxes.append(bbox)
                names.append(name)
                if visualise_confidence:
                    conf = get_conf(object)
                    confs.append(conf)

            if visualise_confidence:
                visualise_img_name_conf(
                    image, names, confs, folder_img, bboxes, dpi=dpi
                )
            else:
                visualise(image, names, folder_img, bboxes, dpi=dpi)


def visualise(image, names, folder_img, bboxes, dpi=80):
    """Visualise a given image in a folder and its corresponding bounding boxes

    Args:
        image ([str]): filename
        names ([str]): name of the annotation box(es)
        folder_img ([str]): folder name containing the image(s)
        bboxes ([list]): list containing 4 tuples (xmin,xmax,ymin,ymax)

    Returns:
        axes_image[matplotlib.image.AxesImage]: matplotlib axes image showing the image and its corresponding bounding box
    """ """"""
    ## standardize all images to have file extension to be .jpg previously and remove all file extension from xml annot earlier
    image_file = f"{image}.jpg"
    image_array = cv2.imread(os.path.join(folder_img, image_file))

    color = (0, 0, 255)
    thickness = 2
    # get matplotlib object to return plot for all in the for loop
    _, ax = plt.subplots(dpi=dpi)
    for (bbox, name,) in zip(bboxes, names):
        bbox = list(map(int, bbox))
        start = (bbox[0], bbox[1])

        end = (bbox[2], bbox[3])

        cv2.rectangle(image_array, start, end, color, thickness)
        cv2.putText(
            image_array,
            name,
            (bbox[0] + 8, bbox[1] + 22),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            color,
        )
    plt.axis("off")
    ax.imshow(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    ax.set_title(image_file)


def visualise_image_with_bbox(folder_img, folder_annot, visualised_list):
    for xml_annotation in visualised_list:
        file = os.path.join(folder_annot, xml_annotation)
        tree = ElementTree.parse(file)
        root = tree.getroot()
        image = get_image_filename(root)
        bboxes = []
        for object in get_object(root):
            name = get_name(object)

            bboxes.append(get_bbox(object))

        visualise(image, name, folder_img, bboxes)

    return len(visualised_list)


#####################
def file_info(dir):
    """Return list of files and count of files"""
    files = os.listdir(dir)
    count = len(os.listdir(dir))
    return files, count


def folder_info(folder_name):
    """Go the subfolders in specified folder and print count of files in each subfolder"""
    folder_info_dict = {}
    for dirpath, dirnames, _ in os.walk(folder_name):
        # dirnames is a empty list, it means the folder only contains files (i.e not a folder with subfolder)
        if not dirnames:
            count = file_info(dirpath)[1]
            folder_info_dict[dirpath] = count

    return folder_info_dict


def get_files_names_with_extension(folder, format):
    """Function returns all files for the particular format"""
    if format == "image":
        exts = (
            ".JPEG",
            "jpeg",
            ".jpg",
            ".tif",
            ".tiff",
            ".bmp",
            ".gif",
            ".png",
            ".raw",
        )
    elif format == "xml":
        exts = "xml"
    else:
        raise Exception('Please input either "image" or "xml" for format argument')

    return [
        f
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and f.endswith(exts)
    ]


def find_images_with_missing_annot(folder_img, folder_annot):
    """Find any images with missing annotations

    Args:
        folder_img ([type]): [description]
        folder_annot ([type]): [description]

    Returns:
        set or False: return the set of images with missing annotation. If no missing annotations, will return False
    """ """
    """
    images = set(
        [
            f.split(".")[0]
            for f in get_files_names_with_extension(folder_img, format="image")
        ]
    )

    annotations = set(
        [f.split(".")[0] for f in get_files_names_with_extension(folder_annot, "xml")]
    )
    diff = images.difference(annotations)

    if diff:
        return diff

    return False


def visualise_from_list(folder_img, visualised_list, ext="", sample_size=4):
    """Visualising image from a provided list of images"""
    if sample_size:
        visualised_list = sample(visualised_list, sample_size)
    for image_filename in visualised_list:
        image_filename = f"{image_filename}.{ext}"
        image_array = plt.imread(os.path.join(folder_img, image_filename))
        _, ax = plt.subplots()
        axes_image = ax.imshow(image_array)

    return axes_image


def unique_xml_names_in_annot_folder_v2(folder_annot):
    """Getting unique names in xml annotation file(s) in the annotation folder"""
    # get all subfolders in main folder
    uniq_xmlnames_in_folders = {}
    for dirpath, dirnames, _ in os.walk(folder_annot):
        if not dirnames:
            uniq_xmlnames_in_folders[dirpath] = set()
            for file in os.listdir(dirpath):
                if file.endswith((".xml")):
                    xml_annotation = os.path.join(dirpath, file)
                    tree = ElementTree.parse(xml_annotation)
                    root = tree.getroot()
                    for object in get_object(root):
                        name = get_name(object)
                        uniq_xmlnames_in_folders[dirpath].add(name)

    return uniq_xmlnames_in_folders


def save_to_eliminated_csv(saved_folder, file, confs_list, conf_threshold):

    saved_path = f"{saved_folder}"
    os.makedirs(saved_path, exist_ok=True)
    saved_path_csv = f"{saved_path}/0.Eliminated_images_{conf_threshold}.csv"

    with open(saved_path_csv, "a", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow([file, *confs_list])
    print(f"Eliminated {file} as the confidence level(s) is/are:{confs_list}")
    return


def xml_folder_remove_conf_thres(folder_annot, saved_path, conf_threshold):
    count_eliminated_xml = 0
    count_of_xml = 0
    for dirpath, dirnames, _ in os.walk(folder_annot):
        if not dirnames:
            for file in os.listdir(dirpath):
                if file.endswith((".xml")):
                    count_of_xml += 1
                    conf_threshold = round(conf_threshold, 2)
                    xml_annotation = os.path.join(dirpath, file)
                    tree = ElementTree.parse(xml_annotation)
                    root = tree.getroot()
                    confs_list = []
                    for object in get_object(root):
                        conf = get_conf(object)
                        confs_list.append(float(conf))

                    os.makedirs(saved_path, exist_ok=True)
                    check_conf_threshold = all(
                        conf >= conf_threshold for conf in confs_list
                    )

                    if check_conf_threshold:
                        tree.write(f"{saved_path}/{file}")
                    else:
                        count_eliminated_xml += 1
                        save_to_eliminated_csv(
                            saved_path, file, confs_list, conf_threshold
                        )
    return count_of_xml, count_eliminated_xml


def counting_img_ext(folder_img):
    """Getting the image file extension(s) in the image folder"""
    img_ext_dict = {}
    for dirpath, dirnames, _ in os.walk(folder_img):
        if not dirnames:
            img_ext_dict[dirpath] = dict()
            for image in os.listdir(dirpath):
                # get the file extension by getting 2nd element
                img_ext = image.split(".")[1]
                if img_ext not in img_ext_dict[dirpath]:
                    img_ext_dict[dirpath][img_ext] = 0
                img_ext_dict[dirpath][img_ext] += 1
    return img_ext_dict


def counting_xml_ext(folder_annot):
    """Getting the image file extension(s) in the annotation folder"""

    xml_ext_dict = {}
    for dirpath, dirnames, _ in os.walk(folder_annot):
        if not dirnames:
            xml_ext_dict[dirpath] = dict()
            for xml_annotation in os.listdir(dirpath):
                file = os.path.join(dirpath, xml_annotation)
                tree = ElementTree.parse(file)
                root = tree.getroot()
                image = get_image_filename(root)
                # if there is a dot, means there is a extension name

                if "." in image:
                    xml_ext = image.split(".")[1]
                    if xml_ext not in xml_ext_dict[dirpath]:
                        xml_ext_dict[dirpath][xml_ext] = 0
                    xml_ext_dict[dirpath][xml_ext] += 1

    return xml_ext_dict


def copy_xmlfolder_with_mapping(src, dest, mapping_dict):
    # get all subfolders in main folder
    subfolders_src = os.listdir(src)
    ## Main Program
    for subfolder in subfolders_src:
        path_to_subfolder = os.path.join(src, subfolder)

        for xml_annotation in os.listdir(path_to_subfolder):

            file = os.path.join(path_to_subfolder, xml_annotation)
            tree = ElementTree.parse(file)
            root = tree.getroot()
            for object in get_object(root):
                to_replace = mapping_dict[subfolder]
                replace_name(object, subfolder, to_replace)

            os.makedirs(f"{dest}/{to_replace}", exist_ok=True)
            # ## writing xml annotation with replaced name to the respective subfolders
            tree.write(f"{dest}/{to_replace}/{xml_annotation}")
    print("Program Completed")


def copy_imagesfolder_with_mapping(src, dest, mapping_dict):
    """Copy image folder with mapping information
    Inputs:
        -src: path to file
        -dest: desired path (not created yet)
        -dict: dictionary containing the mapping for the folder name
    """
    try:
        shutil.copytree(src, dest)
    except FileExistsError:
        print(f"Images folder(s) already exists in {dest}")
        return

    for old_name, new_name in mapping_dict.items():
        old_name_path = os.path.join(dest, old_name)
        new_name_path = os.path.join(dest, new_name)
        try:
            os.rename(old_name_path, new_name_path)
            print(f"{new_name_path} successfully created")
        except FileExistsError:
            ## if the path exist, then rename each files in the file directory with new name
            allfiles = os.listdir(old_name_path)
            for f in allfiles:
                os.rename(
                    os.path.join(old_name_path, f), os.path.join(new_name_path, f)
                )
            os.rmdir(old_name_path)

    return


def convert_imagesfolder_to_jpg(src, dest=None, replace=False):
    if replace and dest:
        raise Exception(
            "replace set to True. Not needed to put dest as function will replace file extension in src folder with .jpg file extension"
        )

    exts = (".JPEG", "jpeg", ".tif", ".tiff", ".bmp", ".gif", ".png", ".raw")
    subfolders = os.listdir(src)

    for subfolder in subfolders:
        path_to_subfolder = os.path.join(src, subfolder)

        for image in os.listdir(path_to_subfolder):
            contains_any_exts = False

            if image.endswith(exts):
                contains_any_exts = True
                filename = image.split(".")[0]
                path_to_image = os.path.join(path_to_subfolder, image)

                if replace:
                    path_to_save = f"{os.path.join(src,subfolder,filename)}.jpg"
                else:
                    try:
                        os.makedirs(os.path.join(dest, subfolder), exist_ok=True)
                    except:
                        raise Exception(
                            "Please input a path for dest since you do not want to replace original folder"
                        )
                    path_to_save = f"{os.path.join(dest,subfolder,filename)}.jpg"

                im = Image.open(path_to_image)
                im.save(path_to_save)
                print(f"Converted {path_to_image} to {path_to_save}")

                if replace:
                    os.remove(path_to_image)

    # if folder contains any exts, will print one statement if not another
    if contains_any_exts:
        print(f"Completed conversion of image extension from {exts} to .jpg")
    else:
        print(f"No file containing image extension of {exts} in {src}. Program Ended")

    return


def remove_xml_ext(read_path, saved_path):
    """Remove extension name in XML annotation in read_path and save files in saved_path"""
    subfolders = os.listdir(read_path)
    for subfolder in subfolders:
        path_to_subfolder = os.path.join(read_path, subfolder)

        for xml_annotation in os.listdir(path_to_subfolder):
            file = os.path.join(path_to_subfolder, xml_annotation)
            tree = ElementTree.parse(file)
            root = tree.getroot()
            image = get_image_filename(root)
            # if there is a dot, means there is a extension name
            if "." in image:
                remove_extension(root)
            os.makedirs(f"{saved_path}/{subfolder}", exist_ok=True)

            tree.write(f"{saved_path}/{subfolder}/{xml_annotation}")
        print(
            f"Completed removal of image file extension for XML annotations in Folder {subfolder}"
        )
    return


def folder_xml_dims(folder_annot):
    """Check the dimensions in xml annotation files in the folder"""
    folder_xml_dims = {}
    for dirpath, dirnames, _ in os.walk(folder_annot):
        if not dirnames:
            folder_xml_dims[dirpath] = set()
            for xml_annotation in os.listdir(dirpath):
                file = os.path.join(dirpath, xml_annotation)
                tree = ElementTree.parse(file)
                root = tree.getroot()
                dim = find_width_height_depth(root)
                folder_xml_dims[dirpath].add(dim)

    return folder_xml_dims


def folder_xml_convert(read_path, saved_path, new_dim):
    subfolders = os.listdir(read_path)
    for subfolder in subfolders:
        path_to_subfolder = os.path.join(read_path, subfolder)

        for xml_annotation in os.listdir(path_to_subfolder):
            file = os.path.join(path_to_subfolder, xml_annotation)
            tree = ElementTree.parse(file)
            root = tree.getroot()
            old_dim = find_dim(root)
            replace_dim(root, new_dim)
            for object in get_object(root):
                new_bbox_list = get_new_bbox_list(object, old_dim, new_dim)
                replace_bbox(object, new_bbox_list)

            # used to create directory recursively, exist_ok-> if already exist leaves dir unaltered
            os.makedirs(f"{saved_path}/{subfolder}", exist_ok=True)
            tree.write(f"{saved_path}/{subfolder}/{xml_annotation}")
        print(
            f"Completed conversion of XML annotations on Folder {subfolder} for new specified dimensions:{new_dim}"
        )
    return


def folder_img_unique_shapes(folder_img):
    """Return the unique shapes in an image folder"""
    rgb_images_set = {}
    grayscale_images_set = {}
    for dirpath, dirnames, _ in os.walk(folder_img):
        if not dirnames:
            rgb_images_set[dirpath] = set()
            grayscale_images_set[dirpath] = set()

            for image_file in os.listdir(dirpath):
                image = os.path.join(dirpath, image_file)
                image_array = plt.imread(image)
                image_array_shape = image_array.shape
                rgb_images_set[dirpath].add(image_array_shape)

                # if grayscale
                if len(image_array_shape) == 2:
                    grayscale_images_set[dirpath].add(image_file)

    return rgb_images_set, grayscale_images_set


def folder_img_convert(
    read_path, saved_path, new_dim=(640, 640), mode="pad_if_small_else_resize_with_pad"
):
    """Convert image to new dimension with desired mode and save it to desired path

    Args:
        read_path ([type]): path to original image
        saved_path ([type]): path to saved files
        new_dim (tuple, optional):new dimension of image.Defaults to (640,640).
        mode (str, optional): either 'resize_with_pad' or 'pad_if_small_else_resize_with_pad' Defaults to "pad_if_small_else_resize_with_pad".

        Modes available
        ---------------
        - resize_with_pad: Apply tf.image.resize_with_pad from tensorflow as per [documentation](https://www.tensorflow.org/api_docs/python/tf/image/resize_with_pad)
        - pad_if_small_else_resize_with_pad:
            - If the original image is smaller than new dimensions (new_dim) then only apply padding up till new dimensions.
            - If the original image is larger than new dimensions (new_dim), then apply tf.image.resize_with_pad from tensorflow as per [documentation](https://www.tensorflow.org/api_docs/python/tf/image/resize_with_pad)
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise Exception()
    modes_available = ["resize_with_pad", "pad_if_small_else_resize_with_pad"]
    if mode not in modes_available:
        raise Exception(
            f"Please enter your desired mode of image transformation from following list: {modes_available}"
        )
    exts = [
        ".JPEG",
        "jpeg",
        ".jpg",
        ".tif",
        ".tiff",
        ".bmp",
        ".gif",
        ".png",
        ".raw",
    ]
    regex_list = [f".+{ext}$" for ext in list(exts)]
    regex_string = "|".join(regex_list)
    new_y, new_x = new_dim
    for dirpath, dirnames, _ in os.walk(read_path):
        if not dirnames:
            for image_file in os.listdir(dirpath):
                if re.match(regex_string, image_file):
                    filename, file_extension = os.path.splitext(image_file)
                    image = os.path.join(dirpath, image_file)
                    try:
                        image_array = plt.imread(image)
                    except Exception:
                        continue
                    ##check if grayscale or colored
                    arr_shape = len(image_array.shape)
                    ## if grayscale
                    if arr_shape == 2:
                        # if grayscale, need to add last dimensions as tf.image.resize_with_pad takes in 3d tensor
                        last_axis = -1
                        image_array = np.expand_dims(image_array, last_axis)
                        dim_to_repeat = 2
                        repeats = 3
                        image_array = np.repeat(image_array, repeats, dim_to_repeat)

                    old_y, old_x, _ = image_array.shape
                    os.makedirs(f"{saved_path}", exist_ok=True)

                    if mode == "resize_with_pad":
                        image_array_resized = tf.image.resize_with_pad(
                            image_array, new_y, new_x
                        )
                        image_array_resized = tf.cast(image_array_resized, tf.uint8)
                        image_array_resized = image_array_resized.numpy()

                    elif mode == "pad_if_small_else_resize_with_pad":

                        # if any part of old image's dimensions is larger than new, do resize with pad
                        if old_y > new_y or old_x > new_x:
                            image_array_resized = tf.image.resize_with_pad(
                                image_array, new_y, new_x
                            )
                            image_array_resized = tf.cast(image_array_resized, tf.uint8)
                            image_array_resized = image_array_resized.numpy()

                            ##do padding only
                        else:
                            diff_y = new_y - old_y
                            diff_x = new_x - old_x
                            top = diff_y // 2
                            bottom = diff_y - top
                            left = diff_x // 2
                            right = diff_x - left
                            image_array_resized = cv2.copyMakeBorder(
                                image_array,
                                top=top,
                                bottom=bottom,
                                left=left,
                                right=right,
                                borderType=cv2.BORDER_CONSTANT,
                                value=0,
                            )

                    saved_image_file = f"{filename}.jpg"
                    plt.imsave(f"{saved_path}/{saved_image_file}", image_array_resized)
            print(
                f"Completed conversion of images on Folder {dirpath} for new specified dimensions:{new_dim}"
            )
    return


def create_error_checking_csv(folder_annot, saved_path, filename):
    os.makedirs(saved_path, exist_ok=True)
    saved_path_csv = f"{saved_path}/Error_checking_{filename}.csv"
    if os.path.isfile(saved_path_csv):
        print(f"{saved_path_csv} already exists")
        return

    with open(saved_path_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Index",
                "Image",
                "Accepted",
                "False Positive",
                "False Negative",
                "Comments",
            ]
        )
    xml_list = [f for f in os.listdir(folder_annot) if f.endswith("xml")]
    files_annot = sorted(xml_list, key=lambda x: split_file(x))
    for index, xml_annotation in enumerate(files_annot):
        image_name = xml_annotation.split(".")[0] + ".jpg"
        with open(saved_path_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [index, image_name,]
            )
    print(f"CSV file saved in {saved_path_csv}")


def get_accepted_results_from_visual_insp(path_to_csv):

    df_error_checking = pd.read_csv(path_to_csv, encoding="unicode_escape")
    df_accepted = df_error_checking[
        df_error_checking.apply(lambda column: column["Accepted"] == 1, axis=1)
    ]
    accepted_list = list(df_accepted["Image"])
    count_of_accepted_list = len(accepted_list)
    return accepted_list, count_of_accepted_list


def get_rejected_results_from_visual_insp(path_to_csv):

    df_error_checking = pd.read_csv(path_to_csv, encoding="unicode_escape")
    df_accepted = df_error_checking[
        df_error_checking.apply(lambda column: column["Accepted"] == 0, axis=1)
    ]
    accepted_list = list(df_accepted["Image"])
    count_of_accepted_list = len(accepted_list)
    return accepted_list, count_of_accepted_list


def copy_accepted_img_and_annot(
    folder_annot, folder_img, dest_annot, dest_img, listing, remove_original=False,
):
    """copy file in listing from folder_annot and folder_img to dest_annot and dest_img

    Args:
        folder_annot ([type]): [description]
        folder_img ([type]): [description]
        dest_annot ([type]): [description]
        dest_img ([type]): [description]
        listing (list): list of the files required for copying
        remove_original (bool, optional): [description]. Defaults to False.
    """
    for file in listing:
        annot = f"{file.split('.')[0]}.xml"
        image = f"{file.split('.')[0]}.jpg"
        path_to_img = os.path.join(folder_img, image)
        dest_to_img = os.path.join(dest_img, image)

        path_to_annot = os.path.join(folder_annot, annot)
        dest_to_annot = os.path.join(dest_annot, annot)

        os.makedirs(dest_img, exist_ok=True)
        os.makedirs(dest_annot, exist_ok=True)
        shutil.copyfile(path_to_img, dest_to_img)
        shutil.copyfile(path_to_annot, dest_to_annot)
        if remove_original:
            os.remove(path_to_img)
            os.remove(path_to_annot)


def create_error_checking_person_annot_csv(folder_annot, saved_path, filename):
    os.makedirs(saved_path, exist_ok=True)
    saved_path_csv = f"{saved_path}//Error_Checking_personin{filename}.csv"
    if os.path.isfile(saved_path_csv):
        print(f"{saved_path_csv} already exists")
        return

    with open(saved_path_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Index",
                "Image",
                "Accepted",
                "False Positive",
                "False Negative",
                "Dump-Out",
                "Comments",
            ]
        )
    xml_list = [f for f in os.listdir(folder_annot) if f.endswith("xml")]
    files_annot = sorted(xml_list, key=lambda x: split_file(x))
    for index, xml_annotation in enumerate(files_annot):
        image_name = xml_annotation.split(".")[0] + ".jpg"
        with open(saved_path_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [index, image_name,]
            )
    print(f"CSV file saved in {saved_path_csv}")


def get_dump_out_results_from_visual_insp(path_to_csv):

    df_error_checking = pd.read_csv(path_to_csv, encoding="unicode_escape")
    df_dump_out = df_error_checking[
        df_error_checking.apply(lambda column: column["Dump-Out"] == 1, axis=1)
    ]
    dump_out_list = list(df_dump_out["Image"])
    count_of_dump_out_list = len(dump_out_list)
    return dump_out_list, count_of_dump_out_list


def replace_xmlfolder_with_mapping(src, dest, mapping_dict):
    """Map object names into the appropriate classes defined in a dictionary"""

    ## Main Program
    for dirpath, dirnames, _ in os.walk(src):
        if not dirnames:
            for file in os.listdir(dirpath):
                if file.endswith((".xml")):
                    path_to_xml_annotation = os.path.join(dirpath, file)
                    tree = ElementTree.parse(path_to_xml_annotation)
                    root = tree.getroot()
                    for object in get_object(root):
                        name = get_name(object)
                        if name in mapping_dict:
                            to_replace = mapping_dict[name]
                            replace_name(object, name, to_replace)

                    os.makedirs(f"{dest}", exist_ok=True)
                    tree.write(f"{dest}/{file}")
                    print(f"Mapped XML Annotations saved in {dest}/{file}")


def xmlfolder_remove_irrelevant_classes(src, dest, irrelevant_classes):
    """Remove irrelevant classes inputted as a list"""
    subfolders_src = os.listdir(src)
    ## Main Program
    for subfolder in subfolders_src:
        path_to_subfolder = os.path.join(src, subfolder)
        for xml_annotation in os.listdir(path_to_subfolder):
            file = os.path.join(path_to_subfolder, xml_annotation)
            tree = ElementTree.parse(file)
            root = tree.getroot()
            for object in get_object(root):
                name = get_name(object)
                if name in irrelevant_classes:
                    root.remove(object)
            os.makedirs(f"{dest}/{subfolder}", exist_ok=True)
            # ## writing xml annotation with replaced name to the respective subfolders
            tree.write(f"{dest}/{subfolder}/{xml_annotation}")
    print("Program Completed")


def create_subfolder_images_and_annotations_with_folder(folder):
    """create subfolder in the folder (Annotations, Images)

    Args:
        folder ([type]): path

    Returns:
        folder/Annotations,folder/Images
    """
    return os.path.join(folder, "Annotations"), os.path.join(folder, "Images")


def split_file(filename):
    first_elem = int(filename.lstrip("n").rstrip(".xml").split("_")[0])
    second_elem = int(filename.lstrip("n").rstrip(".xml").split("_")[1])
    return first_elem, second_elem
