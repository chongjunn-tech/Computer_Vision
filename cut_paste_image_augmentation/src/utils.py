# coding: utf-8
__author__ = "RocketFlash: https://github.com/RocketFlash"

import cv2
import numpy as np
from matplotlib import pyplot as plt
from pascal_voc_writer import Writer
import os
from pathlib import Path
from PIL import Image


def show_image(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title("image")
    plt.show()


def show_image_and_masks(
    image: np.ndarray,
    result_image: np.ndarray,
    result_mask: np.ndarray,
    result_image_anno: np.ndarray,
    is_mask_semantic: bool = False,
):

    f, axarr = plt.subplots(2, 2, figsize=(25, 15))

    axarr[0, 0].imshow(image)
    axarr[0, 1].imshow(result_image)
    if is_mask_semantic:
        result_mask_viz = np.zeros_like(image)
        ids = np.unique(result_mask)
        ids = ids[ids != 0]
        for id_i in ids:
            id_color = list(np.random.choice(range(256), size=3))
            result_mask_viz[result_mask == id_i] = id_color
        result_mask = result_mask_viz

    axarr[1, 0].imshow(result_mask)
    axarr[1, 1].imshow(result_image_anno)

    axarr[0, 0].set_aspect("auto")
    axarr[1, 0].set_aspect("auto")
    axarr[0, 1].set_aspect("auto")
    axarr[1, 1].set_aspect("auto")

    axarr[0, 0].set_xticks([])
    axarr[0, 0].set_yticks([])

    axarr[0, 1].set_xticks([])
    axarr[0, 1].set_yticks([])

    axarr[1, 0].set_xticks([])
    axarr[1, 0].set_yticks([])

    axarr[1, 1].set_xticks([])
    axarr[1, 1].set_yticks([])

    f.subplots_adjust(wspace=0, hspace=0)

    plt.show()


def draw_bboxes(image, bboxes, mask=None, color=(255, 0, 0)):

    result_image = image.copy()
    result_mask = np.zeros_like(result_image)

    if mask is not None:
        ids = np.unique(mask)
        ids = ids[ids != 0]
        for id_i in ids:
            id_color = list(np.random.choice(range(256), size=3))
            result_mask[mask == id_i] = id_color
        cv2.addWeighted(result_mask, 0.4, result_image, 1, 0, result_image)

    for bbox in bboxes:
        if len(bbox) == 5:
            bbox = [int(b) for b in bbox]
            cv2.rectangle(
                result_image, (bbox[1], bbox[2]), (bbox[3], bbox[4]), color, 2
            )

            cv2.putText(
                result_image,
                str(bbox[0]),
                (bbox[1], bbox[2]),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.rectangle(
                result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2
            )

    if mask is None:
        return result_image, None
    else:
        return result_image, result_mask


###NEW FUNCTION
def create_xml_annotation_and_save_image(
    image, bboxes, name=None, mask=None, path="./saved"
):
    """Create and save CAP augmented images to path

    Args:
        image ([type]): [description]
        bboxes ([type]): bboxes can be either in [x1, y1, x2, y2, class_idx] format (if class_idx is input when creating the class) OR [x1, y1, x2, y2,] when class_idx not inputted
        name ([type], optional): name of output (without file extension). Defaults to None.
        mask ([type], optional): [description]. Defaults to None.
        path (str, optional): path to folder to save. Defaults to "./saved".

    Returns:
        if mask=None: result_image, result_mask
        else:
        result_image, result_mask
    
    CAP: Refers to 'cut' and 'paste'
    """

    result_image = image.copy()
    result_mask = np.zeros_like(result_image)
    ## create directory if not exist
    path_to_folder_img = Path(f"{path}/Images")
    path_to_folder_annot = Path(f"{path}/Annotations")
    os.makedirs(path_to_folder_img, exist_ok=True)
    os.makedirs(path_to_folder_annot, exist_ok=True)
    ## save image
    im = Image.fromarray(image)
    path_to_saved_img = os.path.join(path_to_folder_img, f"{name}.jpg")
    path_to_saved_annot = os.path.join(path_to_folder_annot, f"{name}.xml")

    if mask is not None:
        ids = np.unique(mask)
        ids = ids[ids != 0]
        for id_i in ids:
            id_color = list(np.random.choice(range(256), size=3))
            result_mask[mask == id_i] = id_color
        cv2.addWeighted(result_mask, 0.4, result_image, 1, 0, result_image)

    height, width, _ = image.shape
    writer = Writer(path_to_saved_img.rstrip(".jpg"), width, height)
    for bbox in bboxes:
        if len(bbox) == 5:
            bbox = [int(b) for b in bbox]

            writer.addObject(str(bbox[4]), bbox[0], bbox[1], bbox[2], bbox[3])
        else:
            writer.addObject("bag", bbox[0], bbox[1], bbox[2], bbox[3])
    im.save(path_to_saved_img)
    writer.save(path_to_saved_annot)
    print(f"Image saved in {path_to_saved_img}")
    print(f"XML annotation saved in {path_to_saved_annot}")
    if mask is None:
        return result_image, None
    else:
        return result_image, result_mask
