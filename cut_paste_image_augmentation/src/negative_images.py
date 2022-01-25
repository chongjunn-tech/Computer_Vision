from src.pseudo_label_generate import demo_x
from src.Data_Postprocessing import (
    extract_all_files_from_subfolder_to_folder,
    remove_files_in_dir,
)
from .models.onnx_model import ONNXModel
import numpy as np
import os
import shutil


def generate_negative_images(
    images_path,
    output_folder,
    resize_shape=(640, 640),
    eliminated_class=None,
    conf=0.05,
):
    if eliminated_class is None:
        eliminated_class = ["0", "24", "26", "28"]

    path_to_model = os.path.join("src", "exports", "yolox.onnx")
    demo_x(
        ONNXModel(path_to_model, "x"),
        images_path=images_path,
        output_folder=output_folder,
        resize_shape=resize_shape,
        dtype=np.float32,
        xml=True,
        classes_required=eliminated_class,
        confidence_threshold=conf,
        negative_images=True,
    )
    ### removing all files in current directory (these files are unfiltered images)
    remove_files_in_dir(output_folder)
    shutil.rmtree(os.path.join(output_folder, f"1.Annotations-{conf}"))
    ## extract files(i.e. images) from subfolder 1.negative_images to the output_folder
    extract_all_files_from_subfolder_to_folder(output_folder)
