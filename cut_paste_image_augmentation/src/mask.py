import cv2
import numpy as np
from PIL import Image


def otsu_thresholding_mask(img_path):
    img = cv2.imread(img_path, 0)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = np.where(th > 1, 0, 1)
    return mask


def combine_img_mask_into_png(img_path, mask, out_path):
    """
    Convert an object image (.jpg) into a png image with transparent background
    :param img_path: path to object image file in jpg
    :param mask: a numpy array of the object mask (in [0, 1] or [False, True])
    :param out_path: path to the output png file
    :return: nothing (saves the output image into the out_path)
    """

    img = Image.open(img_path)
    arr = np.asarray(img)

    mask = 255 * mask
    mask = np.uint8(mask)
    mask = np.expand_dims(mask, axis=2)

    png = np.concatenate((arr, mask), axis=2)

    out = Image.fromarray(png)

    out.save(out_path)


def generate_trans_image(img_path, out_path):
    mask = otsu_thresholding_mask(img_path)
    combine_img_mask_into_png(img_path, mask, out_path)

