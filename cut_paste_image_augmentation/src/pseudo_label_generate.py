from numpy.core.fromnumeric import resize
from .models.onnx_model import ONNXModel

# from yolov5_processing import *
from .yolox_processing import postprocess, preprocess
import cv2
import numpy as np
from datetime import datetime, date
import os
from pathlib import Path
import csv, time, xml.etree.ElementTree as ET, argparse
from pascal_voc_writer import Writer

IMAGE_EXT = (".jpg", ".jpeg", ".webp", ".bmp", ".png")  # must be tuple


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument(
        "--model_type", type=str, required=True, default="v5", help="v5 or x"
    )
    parser.add_argument(
        "--img_path", default="./generate_data/Unannotated_Images", type=str
    )
    parser.add_argument(
        "--res",
        nargs="+",
        type=int,
        default=[512, 512],
        help="Input resolution. If only 1 argument is provided, it is broadcast to 2 dimensions",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy preprocessing for yolox models",
    )
    parser.add_argument(
        "--xml",
        action="store_true",
        help="run demo with xml annotations saved to saved folder. If turned on, it will not return image(s) with annotation box(es)",
    )
    parser.add_argument(
        "--l",
        "--list",
        nargs="*",
        type=int,
        choices=(list(range(80))),
        help="User to pass in the list of class required from 0-79 (separated with space). Refer to COCO Object Categories for more details. Only applies to xml mode",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="User to pass in the required confidence threshold (default=0.5)",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="./generate_results/Generated_Annotations_Unannotated_Images",
        help="User to pass in desired output folder name(default saved in ./generate_results/Generated_Annotations_Unannotated_Images)",
    )
    parser.add_argument(
        "--flipped_test",
        action="store_true",
        help="Introduce flipped test in the model",
    )
    parser.add_argument(
        "--negative_images", action="store_true", help="negative images",
    )
    return parser


def preprocess_image_folder(
    path, resize_shape=(512, 512), dtype=np.float16, legacy=False, flipped_test=False,
):
    images = []
    meta_data = []
    image_files = []
    flipped_images = []
    for image_file in Path(path).iterdir():
        if not str(image_file).endswith(IMAGE_EXT):
            pass
        else:
            image = cv2.imread(str(image_file))
            raw_image = image.copy()
            image, ratio = preprocess(image, resize_shape, legacy=legacy)
            image = np.expand_dims(image, axis=0)
            image = np.array(image, dtype=dtype)
            images.append(image)
            meta_data.append(
                {"raw_img": raw_image, "ratio": ratio, "test_size": resize_shape}
            )
            head_tail = os.path.split(image_file)
            tail = head_tail[1]
            image_files.append(tail)

            if flipped_test:
                tmp_image = raw_image.copy()
                flipped_image = np.fliplr(tmp_image)
                flipped_image, ratio = preprocess(
                    flipped_image, resize_shape, legacy=legacy
                )
                flipped_image = np.expand_dims(flipped_image, axis=0)
                flipped_image = np.array(flipped_image, dtype=dtype)
                flipped_images.append(flipped_image)

    if flipped_test:
        return (images, flipped_images, meta_data, image_files)
    else:
        return (images, meta_data, image_files)


def generate_xml_annotation(
    output_bboxes,
    names,
    conf_values,
    output_folder,
    image_file,
    resize_shape,
    images_path,
    confidence_threshold=0.5,
):
    """Generate XML annotation from a given output bounding box(es), name(s) of object and the image file

    Args:
        output_bboxes (list): output bounding boxes generated from the model
        names (list): list of name(s) generated from the model
        conf_values (list): list of confidence value(s)
        output_folder (str): Folder to save XML annotation
        image_file (str): filename of the image
        resize_shape (tuple): shape in (height,width)
        images_path (str): path to the folder containing the images
        confidence_threshold (float, optional): the confidence threshold used for prediction used by the model. Defaults to 0.5.

    Returns:
        xml_count (int): count of xml generated
        
    """
    xml_count = 0

    file_without_ext = image_file.split(".")[0]
    saved_path = f"{output_folder}/1.Annotations-{confidence_threshold}"
    image_path = os.path.join(images_path, file_without_ext)
    os.makedirs(saved_path, exist_ok=True)
    saved_xml_path = f"{os.path.join(saved_path, file_without_ext)}.xml"
    h, w = resize_shape
    writer = Writer(image_path, w, h)
    for output_box, name in zip(output_bboxes, names):
        x0, y0, x1, y1 = output_box
        writer.addObject(name, x0, y0, x1, y1)

    writer.save(saved_xml_path)
    ## using Elementtree to create a confidence as cannot be created with Pascal Writer
    tree = ET.parse(saved_xml_path)
    root = tree.getroot()
    pseudo = ET.SubElement(root, "pseudo")
    pseudo.text = str(1)
    for bndbox, conf in zip(root.findall("./object/bndbox"), conf_values):
        confidence = ET.SubElement(bndbox, "confidence")
        confidence.text = conf

    tree.write(saved_xml_path)
    xml_count += 1
    print(f"XML Annotatation saved in {saved_xml_path}")

    return xml_count


def save_to_no_detection_csv(output_folder, image_file, confidence_threshold=0.5):
    """Save image with no detection returned by model to {output_folder}/Annotations-{confidence_threshold}/1.no_detection.csv
    First column of csv file contains the image(s) with no returned detection
    Only applies when user select --xml
    """
    saved_path = f"{output_folder}/1.Annotations-{confidence_threshold}"
    os.makedirs(saved_path, exist_ok=True)
    saved_path_csv = f"{saved_path}/1.no_detection.csv"
    with open(saved_path_csv, "a", newline="") as (f):
        writer = csv.writer(f)
        writer.writerow([image_file])
    print(f"No detection return for {image_file}")
    return saved_path_csv


def save_to_failed_flipped_test_csv(
    output_folder, image_file, confidence_threshold=0.5
):
    saved_path = f"{output_folder}/1.Annotations-{confidence_threshold}"
    os.makedirs(saved_path, exist_ok=True)
    saved_path_csv = f"{saved_path}/0.failed_flipped_test.csv"
    with open(saved_path_csv, "a", newline="") as (f):
        writer = csv.writer(f)
        writer.writerow([image_file])
    print(f"Failed flipped test for {image_file}")


def save_to_model_summary(
    output_folder,
    confidence_threshold,
    classes_required,
    no_detection_counts,
    xml_counts_generated_or_left,
    images_or_XML_analyzed_count,
    failed_flipped_test_counts=0,
    total_removed_xml_count=None,
    comments=None,
):
    """Save the overall summary of the model results into {output_folder}//Model_Summary.csv
    Only applies when user select --xml

    """
    if total_removed_xml_count is None:
        total_removed_xml_count = no_detection_counts + failed_flipped_test_counts
    saved_path_csv = f"{output_folder}/Model_Summary.csv"
    if not os.path.isfile(saved_path_csv):
        with open(saved_path_csv, "a", newline="") as (f):
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Confidence Interval",
                    "Classes inputted",
                    "Count of no detection",
                    "Count of failed flipped test",
                    "Count of Removed XML files",
                    "Count of XML File generated/left",
                    "Count of Images/XML Analyzed",
                    "Comments",
                ]
            )
    with open(saved_path_csv, "a", newline="") as (f):
        writer = csv.writer(f)
        writer.writerow(
            [
                confidence_threshold,
                classes_required,
                no_detection_counts,
                failed_flipped_test_counts,
                total_removed_xml_count,
                xml_counts_generated_or_left,
                images_or_XML_analyzed_count,
                comments,
            ]
        )


def save_negative_images(
    csv_file, path_to_image_folder, path_to_negative_images_folder
):
    import pandas as pd
    import shutil

    os.makedirs(path_to_negative_images_folder, exist_ok=True)

    df = pd.read_csv(csv_file, header=None)
    list_of_no_detection = list(df[0])
    for image in list_of_no_detection:
        path_to_image = os.path.join(path_to_image_folder, image)
        dest_to_image = os.path.join(path_to_negative_images_folder, image)
        shutil.copyfile(path_to_image, dest_to_image)


def vis_batch(
    outputs,
    meta_data,
    resize_shape,
    xml=False,
    image_files=None,
    output_folder=None,
    images_path=None,
    classes_required=None,
    confidence_threshold=0.5,
    negative_images=False,
):
    if output_folder == "./generate_results":
        output_folder = os.path.join(output_folder, str(date.today()))
    Path(output_folder).mkdir(exist_ok=True, parents=True)

    no_detection_counts = 0
    xml_counts = 0
    images_count = len(image_files)

    for index, (output, data, image_file) in enumerate(
        zip(outputs, meta_data, image_files)
    ):
        if xml:
            result = postprocess(
                output,
                data,
                class_names=[str(x) for x in range(80)],
                create_visualization=False,
                confidence_threshold=confidence_threshold,
            )
            try:
                output_bboxes, names, conf_values = result
            except:
                output_bboxes = names = conf_values = None

            ## if classes required need prefilter the result based on if it is as per classes required
            if classes_required and output_bboxes and names and conf_values:
                bboxes, labels, scores = result
                result = list(
                    filter(
                        lambda x: str(x[1]) in classes_required,
                        zip(bboxes, labels, scores),
                    )
                )
                result = list(zip(*result))

                # reassign variables to only results that are inside class_required
                if result:
                    output_bboxes, names, conf_values = result
                else:
                    save_to_no_detection_csv(
                        output_folder,
                        image_file,
                        confidence_threshold=confidence_threshold,
                    )
                    no_detection_counts += 1
                    continue

            else:
                saved_path_csv = save_to_no_detection_csv(
                    output_folder,
                    image_file,
                    confidence_threshold=confidence_threshold,
                )
                no_detection_counts += 1
                continue

            xml_count = generate_xml_annotation(
                output_bboxes,
                names,
                conf_values,
                output_folder,
                image_file,
                resize_shape,
                images_path,
                confidence_threshold=confidence_threshold,
            )
            xml_counts += xml_count

        else:
            vis_image = postprocess(
                output,
                data,
                class_names=[str(x) for x in range(80)],
                confidence_threshold=confidence_threshold,
            )
            cv2.imwrite(
                str(os.path.join(output_folder, f"vis_{image_file}")), vis_image
            )

    if xml:
        save_to_model_summary(
            output_folder,
            confidence_threshold,
            classes_required,
            no_detection_counts,
            xml_counts,
            images_count,
        )
    if negative_images:
        save_negative_images(
            csv_file=saved_path_csv,
            path_to_image_folder=images_path,
            path_to_negative_images_folder=os.path.join(
                output_folder, "1.negative_images"
            ),
        )


def vis_batch_flipped_image_test(
    outputs,
    meta_data,
    resize_shape,
    xml=False,
    image_files=None,
    output_folder=None,
    images_path=None,
    classes_required=None,
    confidence_threshold=0.5,
    flipped_images_outputs=None,
    negative_images=False,
):
    """Perform vis_batch with flipped_test

    *flipped_test: Refers to flipping the image horizontally and testing if the number of generated annotation(s) is/are the same number as generated annotation(s) for non-flipped image
    Args:
        outputs (list): list of output from model
        meta_data (list): list of meta_data from model
        resize_shape (tuple): resize shape in (height,width)
        xml (bool, optional): If True will generate xml annotation instead of generate bounding box onto images. Defaults to False.
        image_files (list, optional): list of image files. Defaults to None.
        output_folder (str, optional): path to output folder. Defaults to None.
        images_path (str, optional): path to unannotated image(s). Defaults to None.
        classes_required (list, optional): list of classes_required (in string). Defaults to None.
        confidence_threshold (float, optional): confidence threshold needed by the model. Defaults to 0.5.
        flipped_images_outputs (list, optional): list of flipped_images_outputs. Defaults to None.
    """
    if output_folder == "./generate_results":
        output_folder = os.path.join(output_folder, str(date.today()))
    Path(output_folder).mkdir(exist_ok=True, parents=True)

    no_detection_counts = 0
    xml_counts = 0
    failed_flipped_test_counts = 0
    images_count = len(image_files)

    for index, (output, flipped_images_output, data, image_file) in enumerate(
        zip(outputs, flipped_images_outputs, meta_data, image_files)
    ):
        if xml:
            result = postprocess(
                output,
                data,
                class_names=[str(x) for x in range(80)],
                create_visualization=False,
                confidence_threshold=confidence_threshold,
            )

            flipped_image_result = postprocess(
                flipped_images_output,
                data,
                class_names=[str(x) for x in range(80)],
                create_visualization=False,
                confidence_threshold=confidence_threshold,
            )

            try:
                output_bboxes, names, conf_values = result
            except:
                output_bboxes = names = conf_values = None
            try:
                _, flipped_image_names, _, = flipped_image_result
            except:
                flipped_image_names = None

            if classes_required and output_bboxes and names and conf_values:
                ## convert classes required to list of strings
                if flipped_image_names:
                    classes_required = list(map(str, classes_required))
                    bboxes, labels, scores = result
                    result = list(
                        filter(
                            lambda x: x[1] in classes_required,
                            zip(bboxes, labels, scores),
                        )
                    )
                    result = list(zip(*result))
                    flipped_image_names = [
                        name for name in flipped_image_names if name in classes_required
                    ]

                    # reassign variables to only results that are inside class_required
                    if result:
                        output_bboxes, names, conf_values = result
                    else:
                        save_to_no_detection_csv(
                            output_folder,
                            image_file,
                            confidence_threshold=confidence_threshold,
                        )
                        no_detection_counts += 1
                        continue
                else:
                    save_to_failed_flipped_test_csv(
                        output_folder,
                        image_file,
                        confidence_threshold=confidence_threshold,
                    )
                    failed_flipped_test_counts += 1
                    continue
            else:
                save_to_no_detection_csv(
                    output_folder,
                    image_file,
                    confidence_threshold=confidence_threshold,
                )
                no_detection_counts += 1
                continue
            if output_bboxes and names and conf_values:

                if len(output_bboxes) != len(flipped_image_names):
                    save_to_failed_flipped_test_csv(
                        output_folder,
                        image_file,
                        confidence_threshold=confidence_threshold,
                    )
                    failed_flipped_test_counts += 1
                    continue
                else:
                    pass

                xml_count = generate_xml_annotation(
                    output_bboxes,
                    names,
                    conf_values,
                    output_folder,
                    image_file,
                    resize_shape,
                    images_path,
                    confidence_threshold=confidence_threshold,
                )
                xml_counts += xml_count

            else:
                saved_path_csv = save_to_no_detection_csv(
                    output_folder,
                    image_file,
                    confidence_threshold=confidence_threshold,
                )
                no_detection_counts += 1
                continue

        else:
            vis_image = postprocess(
                output,
                data,
                class_names=[str(x) for x in range(80)],
                confidence_threshold=confidence_threshold,
            )
            cv2.imwrite(
                str(os.path.join(output_folder, f"vis_{image_file}")), vis_image
            )

    if xml:
        save_to_model_summary(
            output_folder,
            confidence_threshold,
            classes_required,
            no_detection_counts,
            xml_counts,
            images_count,
            failed_flipped_test_counts=failed_flipped_test_counts,
        )

    if negative_images:
        save_negative_images(
            csv_file=saved_path_csv,
            path_to_image_folder=images_path,
            path_to_negative_images_folder=os.path.join(
                output_folder, "1.negative_images"
            ),
        )


def demo_x(
    model,
    images_path="./generate_data",
    output_folder="./generate_results",
    resize_shape=(512, 512),
    dtype=np.float16,
    legacy=False,
    xml=False,
    classes_required=False,
    confidence_threshold=0.5,
    flipped_test=False,
    negative_images=False,
):
    assert len(resize_shape) == 1 or len(resize_shape) == 2
    if len(resize_shape) == 1:
        resize_shape = (resize_shape[0], resize_shape[0])

    if flipped_test:
        images, flipped_images, meta_data, image_files = preprocess_image_folder(
            images_path,
            resize_shape=resize_shape,
            dtype=dtype,
            legacy=legacy,
            flipped_test=flipped_test,
        )
    else:
        images, meta_data, image_files = preprocess_image_folder(
            images_path, resize_shape=resize_shape, dtype=dtype, legacy=legacy
        )
    outputs = []
    flipped_images_outputs = []
    for image in images:
        start_time = time.time()
        output = model.forward(image)
        print(f"Infer time: {round(time.time() - start_time, 4)}s")
        outputs.append(output)

    if flipped_test:
        for flipped_image in flipped_images:
            start_time = time.time()
            flipped_image_output = model.forward(flipped_image)
            print(f"Infer time: {round(time.time() - start_time, 4)}s")
            flipped_images_outputs.append(flipped_image_output)

    model.close()

    if flipped_test:
        vis_batch_flipped_image_test(
            outputs,
            meta_data,
            resize_shape,
            xml,
            image_files,
            output_folder=output_folder,
            images_path=images_path,
            classes_required=classes_required,
            confidence_threshold=confidence_threshold,
            flipped_images_outputs=flipped_images_outputs,
            negative_images=negative_images,
        )
    else:
        vis_batch(
            outputs,
            meta_data,
            resize_shape,
            xml,
            image_files,
            output_folder=output_folder,
            images_path=images_path,
            classes_required=classes_required,
            confidence_threshold=confidence_threshold,
            negative_images=negative_images,
        )


###################### YOLOv5 ##########################
def rknn_post_process(outputs, img_size):
    # full post process
    input0_data = outputs[1].transpose(0, 1, 4, 2, 3)
    input1_data = outputs[2].transpose(0, 1, 4, 2, 3)
    input2_data = outputs[3].transpose(0, 1, 4, 2, 3)

    input0_data = input0_data.reshape(*input0_data.shape[1:])
    input1_data = input1_data.reshape(*input1_data.shape[1:])
    input2_data = input2_data.reshape(*input2_data.shape[1:])

    input_data = []
    input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

    boxes, classes, scores = rknn_yolov5_post_process_full(input_data, img_size)
    return boxes, classes, scores


def rknn_vis_save(img, boxes, scores, classes, output_folder, img_name):
    img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if boxes is not None:
        rknn_draw(img_1, boxes, scores, classes)
    print("Saving result to", output_folder + "/" + img_name)
    if not cv2.imwrite(output_folder + "/" + img_name.split("/")[-1], img_1):
        raise Exception("Could not write image")


def onnx_vis_save(img, im0s, pred, output_folder, img_name):
    for det in pred:
        im0 = im0s.copy()
        annotator = Annotator(im0, line_width=1, example=str(CLASSES))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = f"{CLASSES[c]} {conf:.2f}"
                annotator.box_label(xyxy, label, color=colors(c, True))
        if not cv2.imwrite(output_folder + "/" + img_name.split("/")[-1], im0):
            raise Exception("Could not write image")


def demo_v5(model, model_type, test_dir, test_save, resize_shape=512):
    output_folder = test_save + "/YOLOv5_" + str(datetime.now())
    print("Inference results will be saved to:", output_folder)
    Path(output_folder).mkdir(exist_ok=True, parents=True)
    if model_type == "onnx":
        dataset = LoadImages(test_dir, img_size=resize_shape[0], stride=64, auto=False)
        for path, img, im0s, vid_cap in dataset:
            img = onnx_preprocess(img)
            start_time = time.time()
            pred = model.forward(img)
            print(f"Infer time: {round(time.time() - start_time, 4)}s")
            pred = torch.tensor(pred)
            pred = onnx_non_max_suppression(pred, 0.25, 0.45)
            onnx_vis_save(img, im0s, pred, output_folder, path.split("/")[-1])
        model.close()
    else:
        for img in Path(test_dir).iterdir():
            img_name = str(img)
            print("Performing inference on:", img_name)
            # ignore non-image files:
            if not img_name.endswith(IMAGE_EXT):
                continue

            img = cv2.imread(str(img))
            img, ratio, (dw, dh) = rknn_letterbox(
                img, new_shape=(*resize_shape, *resize_shape)
            )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Inference
            print("--> Running model")
            start_time = time.time()
            outputs = model.forward(img)
            print(f"Infer time: {round(time.time() - start_time, 4)}s")
            boxes, classes, scores = rknn_post_process(outputs, *resize_shape)
            rknn_vis_save(img, boxes, scores, classes, output_folder, img_name)


if __name__ == "__main__":
    # colors = Colors()

    args = make_parser().parse_args()

    engine = args.model_path.split(".")[-1]

    if engine == "onnx":
        path_to_model = os.path.join("./exports", args.model_path)
        if not os.path.exists(path_to_model):
            print("Model does not exist")
            exit(-1)
        model = ONNXModel(path_to_model, args.model_type)

    if args.l:
        # convert classes_required to a list of strings as the model's output is in list of strings
        classes_required = list(map(str, args.l))
    else:
        classes_required = None

    if args.model_type == "v5":
        demo_v5(model, engine, args.test_dir, args.test_save, args.res)
    elif args.model_type == "x":
        demo_x(
            model,
            images_path=(args.img_path),
            output_folder=(args.results_path),
            resize_shape=(args.res),
            dtype=np.float32,
            legacy=(args.legacy),
            xml=(args.xml),
            classes_required=classes_required,
            confidence_threshold=(args.conf),
            flipped_test=args.flipped_test,
            negative_images=args.negative_images,
        )

