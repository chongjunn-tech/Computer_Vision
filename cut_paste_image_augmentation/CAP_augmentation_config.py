from easydict import EasyDict

config = EasyDict(
    dict(
        precut_images="precut_images",
        cut_images="cut_images",
        paste_images="paste_images",
        paste_videos="paste_videos",
        resize_shape=(640, 640),
        eliminated_class=["0", "24", "26", "28"],
        confidence=0.01,
        CAP_folder="saved",
    )
)

