import os
import cv2
import numpy as np
import xml.etree.ElementTree as ElementTree
from pascal_voc_writer import Writer
import shutil
import matplotlib.pyplot as plt
from .xml_change import *
from pathlib import Path


def check_four_corners(grayscale_array):
    """Check if four corners of a grayscale array is 0

    Args:
        grayscale_array ([type]): [description]

    Returns:
        [type]: [description]
    """
    return (
        grayscale_array[0][0]
        == grayscale_array[0][-1]
        == grayscale_array[-1][0]
        == grayscale_array[-1][-1]
        == 0
    )


def get_unpadded_region_coordinate(path_to_img):
    """Get the top left and bottom right corner of unpadded region

    Args:
        path_to_img ([type]): [description]

    Returns:
        [type]: [description]
    """
    img = cv2.imread(path_to_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if check_four_corners(gray):

        arr = np.argwhere(gray >= 1)
        z = sorted(arr, key=lambda t: t[0] + t[1])
        y1, x1 = z[0]
        y2, x2 = z[-1]

    else:
        return False

    return (x1, y1), (x2, y2)


def folder_xml_limit_bbox_to_unpadded_region(
    folder_annot, folder_img, dest_folder_annot, listing=None
):
    """Limit the generated annotation by deep learning model to be restricted to the unpadded region

    Args:
        folder_annot ([type]): [description]
        folder_img ([type]): [description]
        dest_folder_annot ([type]): [description]
    """
    ## if user put in listing, will look through listing else, will look through whole folder_img
    if listing:
        images_to_look_through = [f.split(".")[0] + ".jpg" for f in listing]
    else:
        images_to_look_through = os.listdir(folder_img)

    for img in images_to_look_through:
        path_to_img = os.path.join(folder_img, img)
        filename = img.split(".")[0]
        xml_annotation = f"{filename}.xml"
        path_to_xml_annotation = os.path.join(folder_annot, xml_annotation)
        os.makedirs(dest_folder_annot, exist_ok=True)
        saved_xml_annot = f"{dest_folder_annot}/{xml_annotation}"
        if get_unpadded_region_coordinate(path_to_img):

            (max_x1, max_y1), (max_x2, max_y2) = get_unpadded_region_coordinate(
                path_to_img
            )
            tree = ElementTree.parse(path_to_xml_annotation)
            root = tree.getroot()
            for object in get_object(root):
                bbox = tuple(map(int, get_bbox(object)))
                xmin, ymin, xmax, ymax = bbox
                x1 = max(xmin, max_x1)
                y1 = max(ymin, max_y1)
                x2 = min(xmax, max_x2)
                y2 = min(ymax, max_y2)
                new_bbox = x1, y1, x2, y2
                replace_bbox(object, new_bbox)

            tree.write(saved_xml_annot)

        else:

            shutil.copyfile(path_to_xml_annotation, saved_xml_annot)
    print(
        f"Refined annotations to unpadded regions for {folder_annot} saved to :{dest_folder_annot}"
    )


def folder_img_generate_blank_annotation_for_negative_images(folder_img, folder_annot):

    os.makedirs(folder_annot, exist_ok=True)
    for img in os.listdir(folder_img):
        img_without_ext = img.split(".")[0]
        path_to_image = os.path.join(folder_img, img)
        annot = f"{img_without_ext}.xml"
        path_to_annot = os.path.join(folder_annot, annot)
        img_array = plt.imread(path_to_image)
        H, W, _ = img_array.shape
        writer = Writer(path_to_image, W, H)
        writer.save(path_to_annot)


def extract_all_files_from_subfolder_to_folder(
    folder_path, prepend_folder_name_in_filename=False
):
    """Extract files from subfolders in folder to the folder itself and delete empty subfolder

    Args:
        folder_path ([type]): filepath
        prepend_folder_name_in_filename (bool, optional): prepend folder name to the actual filename if True. Defaults to False.
    """
    subfolders = [
        folder
        for folder in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, folder))
    ]
    for sub in subfolders:
        path_to_sub = os.path.join(folder_path, sub)
        for filename in os.listdir(path_to_sub):
            src = os.path.join(folder_path, sub, filename)
            if os.path.isfile(src):
                ## if true, then prepend folder name to the actual filename and also change the filename in xml annotation
                if prepend_folder_name_in_filename:
                    new_filename = f"{sub}_{filename}"
                    if filename.endswith(".xml"):
                        tree = ElementTree.parse(src)
                        root = tree.getroot()
                        root.find("filename").text = new_filename.rstrip(".xml")
                        tree.write(src)

                    filename = new_filename
                dst = os.path.join(folder_path, filename)
                shutil.move(src, dst)

        os.rmdir(path_to_sub)


def move_files(abs_dirname, filesize_limit):
    """Move files into subdirectories if it contains images more than filesize_limit

    Args:
        abs_dirname ([type]): filepath
        filesize_limit (int): the limit of files in the folder
    """
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
    files = [
        os.path.join(abs_dirname, f)
        for f in os.listdir(abs_dirname)
        if f.endswith(exts)
    ]
    count_of_img = len(files)
    print(f"Folder ({abs_dirname}):{count_of_img} images")
    ## if the count of image
    if count_of_img > filesize_limit:
        print(
            f"Folder ({abs_dirname}) will be split into subfolders containing {filesize_limit} images each"
        )
        i = 0
        path = os.path.normpath(abs_dirname)
        try:
            back_path = os.path.join(*path.split(os.sep)[:-1])
        except:
            back_path = "."
        prepend = path.split(os.sep)[-1]
        for f in files:
            # create new subdir if necessary
            if i % filesize_limit == 0:
                batch_no = i // filesize_limit + 1
                subdir_name = os.path.join(back_path, f"{prepend}_Batch_{batch_no}")
                os.makedirs(subdir_name, exist_ok=True)

            # move file to current dir
            f_base = os.path.basename(f)
            shutil.move(f, os.path.join(subdir_name, f_base))
            i += 1
        # remove original folder (now empty)
        os.rmdir(abs_dirname)
        print(
            f"Completed batch size conversion for folder ({abs_dirname}) with {batch_no} batches"
        )
        return
    else:
        pass


def xml_folder_to_append(
    folder_annot, folder_annot_to_append, saved_path, class_desired
):
    """Append generated annotation to the original annotation file"""
    os.makedirs(saved_path, exist_ok=True)
    for xml_annotation in os.listdir(folder_annot):
        if xml_annotation.endswith(".xml"):
            add_file = os.path.join(folder_annot, xml_annotation)
            tree = ElementTree.parse(add_file)
            add_root = tree.getroot()

            file_to_append = os.path.join(folder_annot_to_append, xml_annotation)

            bboxes_to_append = []
            names_to_append = []

            if os.path.isfile(file_to_append):
                tree_to_append = ElementTree.parse(file_to_append)
                root_to_append = tree_to_append.getroot()
                for object_in_root_to_append in get_object(root_to_append):
                    name_to_append = get_name(object_in_root_to_append)
                    bbox_to_append = get_bbox(object_in_root_to_append)
                    bboxes_to_append.append(bbox_to_append)
                    names_to_append.append(name_to_append)

                for object in get_object(add_root):
                    add_bbox = get_bbox(object)
                    add_name = get_name(object)

                    if add_bbox not in bboxes_to_append and add_name in class_desired:
                        ET_object = ElementTree.SubElement(root_to_append, "object")
                        ET_name = ElementTree.SubElement(ET_object, "name")
                        ET_bndbox = ElementTree.SubElement(ET_object, "bndbox")
                        ET_xmin = ElementTree.SubElement(ET_bndbox, "xmin")
                        ET_ymin = ElementTree.SubElement(ET_bndbox, "ymin")
                        ET_xmax = ElementTree.SubElement(ET_bndbox, "xmax")
                        ET_ymax = ElementTree.SubElement(ET_bndbox, "ymax")
                        ET_name.text = add_name
                        ET_xmin.text = add_bbox[0]
                        ET_ymin.text = add_bbox[1]
                        ET_xmax.text = add_bbox[2]
                        ET_ymax.text = add_bbox[3]
                    ElementTree.indent(tree_to_append, space="\t", level=0)
                    tree_to_append.write(
                        f"{saved_path}/{xml_annotation}", encoding="utf-8"
                    )


def remove_files_in_dir(folder_path):
    """Remove all files in the folder (Only file Not Folder)

    Args:
        folder_path (str): relative path to folder
    """
    ### removing all images in current directory
    p = Path(folder_path)
    for file in p.iterdir():
        if file.is_file():
            Path.unlink(file)
