import os
import cv2
from pathlib import Path
from src.cap_aug import CAP_AUG
from src.utils import create_xml_annotation_and_save_image
from src.video_processing import video_processing
from src.mask import generate_trans_image
import argparse
from CAP_augmentation_config import config as cfg


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos",
        action="store_true",
        help="If passed, will process all the videos placed in paste_videos into frames and save in folder (past_images)",
    )
    parser.add_argument(
        "--negative_images",
        action="store_true",
        help="If passed, will process all the videos placed in paste_images into negative images (i.e. ONNX Model does not detect any object(s) listed in eliminated class(es)",
    )
    parser.add_argument(
        "--precut_images",
        action="store_true",
        help="If passed, will transform the images in precut_images into mask image",
    )
    return parser


def gen_xml_and_image_from_trans_folder(
    cut_folder="cut_images",
    paste_folder="paste_images",
    n_objects_range=None,
    offset=40,
    class_idx=None,
    CAP_folder="./saved",
):
    """Generate cut and paste augmentations.
    Cut images from folder cut_images
    paste images from folder paste_images

    Args:
        cut_folder (str): path to folder containing cut images
        paste_folder (str): path to folder containing paste images
        n_objects_range (list,optional): [min, max] number of objects. Defaults to [1,2]
        offset(int,optional): offset from the border to prevent cases where object is near the image's border
        class_idx (int, optional): Defaults to None. If None, generated xml annotation name will be 'bag' as per utils.py. If passed, the name in xml annotation will be class_idx (int)
    """
    ## place this assignment here as not advisable to put list as default parameters
    ## if user input None, then place these default parameters
    if n_objects_range is None:
        n_objects_range = [1, 2]

    ## Load Data
    cut_folder_path = Path(cut_folder)
    paste_folder_path = Path(paste_folder)
    ### Load paste images
    paste_images_names = list(paste_folder_path.glob("**/*"))

    ## Load cut images
    cut_images = sorted(list(cut_folder_path.glob("*.png")))
    ## check if folder cut_images contains any png images
    assert len(cut_images) > 0, f"No png image(s) found in Folder:{cut_folder}"

    for paste_image_name in paste_images_names:
        ##get original image name without image extension behind
        name_without_ext = os.path.split(paste_image_name)[-1].split(".")[0]
        ## append '_CAP' to the original name with extension
        name_CAP = f"{name_without_ext}_CAP"
        image = cv2.imread(str(paste_image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ## determining the range by the image's shape
        max_y, max_x, _ = image.shape
        x_range = [offset, max_x - offset]
        # limit y_range so that the object can be near the floor/ground
        y_range = [max_y - (1.2 * offset), max_y - offset]
        min_h = int(max_y / 6)
        max_h = 1.1 * min_h
        h_range = [min_h, max_h]
        ## change the parameters here to alter behavior of cut and paste
        cap_aug_pixels = CAP_AUG(
            cut_images,
            bev_transform=None,
            n_objects_range=n_objects_range,
            h_range=h_range,
            x_range=x_range,
            y_range=y_range,
            class_idx=class_idx,
            image_format="rgb",
        )
        result_image, result_coords, _, instance_mask = cap_aug_pixels(image)
        ##save CAP images and xml annnotation
        create_xml_annotation_and_save_image(
            result_image,
            result_coords,
            name=name_CAP,
            mask=instance_mask,
            path=CAP_folder,
        )


if __name__ == "__main__":

    args = make_parser().parse_args()

    if args.precut_images:
        precut_images = Path(cfg.precut_images).glob("*")

        for precut_image in precut_images:

            image_name = os.path.split(precut_image)[-1].strip(".jpg") + "_masked"
            os.makedirs("cut_images", exist_ok=True)
            saved_path = os.path.join("cut_images", f"{image_name}.PNG")
            generate_trans_image(str(precut_image), saved_path)

    if args.videos:
        videos = Path(cfg.paste_videos).glob("*")
        ## Looping through videos in PASTE_VIDEOS
        for video in videos:
            ##convert from  class pathlib.WindowsPath to class string or else cv2.Videocapture will not work
            video = str(video)
            video_processing(video, cfg.paste_images, remove_blur=True)

    if args.negative_images:
        try:
            from src.negative_images import generate_negative_images
        except:
            raise Exception(
                "Install the required dependencies listed in requirements.txt"
            )
        generate_negative_images(
            images_path=cfg.paste_images,
            output_folder=cfg.paste_images,
            resize_shape=cfg.resize_shape,
            eliminated_class=cfg.eliminated_class,
            conf=cfg.confidence,
        )

    gen_xml_and_image_from_trans_folder(
        cut_folder=cfg.cut_images,
        paste_folder=cfg.paste_images,
        CAP_folder=cfg.CAP_folder,
    )
