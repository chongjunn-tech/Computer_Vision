import os
from pathlib import Path
import cv2


path_to_video = "preprocessed/South_Beach.mp4"
dest_folder = "postprocessed"

## This method of checking blur may not be very effective
## reference:https://www.analyticsvidhya.com/blog/2020/09/how-to-perform-blur-detection-using-opencv-in-python/
def blur_detection(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()

    blur = fm < threshold
    return blur


def video_processing(path_to_video, dest_folder, remove_blur=False):
    cam = cv2.VideoCapture(path_to_video)
    idx = 0
    ##get the video_name without extension
    video_name = os.path.split(path_to_video)[-1].split(".")[0]
    os.makedirs(dest_folder, exist_ok=True)

    while True:

        # reading from frame
        ret, frame = cam.read()

        if ret:
            # if video is still left continue creating images
            name = os.path.join(dest_folder, video_name + "_frame_" + str(idx) + ".jpg")

            if remove_blur:
                check_dispose_image = blur_detection(frame)
            else:
                check_dispose_image = False

            # writing the extracted images
            if idx % 10 == 0 and not check_dispose_image:
                print("Creating..." + name)
                cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            idx += 1
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()
    print("Video Processing Program ended")


if __name__ == "__main__":
    video_processing(path_to_video, dest_folder)

