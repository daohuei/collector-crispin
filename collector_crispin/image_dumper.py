import numpy as np
import os
import cv2
from motion_marmot.utils import video_utils
from motion_marmot.advanced_motion_filter import (
    AdvancedMotionFilter,
)

# Importing Image module from PIL package
from PIL import Image


class ImageFrame:
    def __init__(self):
        self.black_frame = np.zeros((180, 320, 3), np.uint8)
        self.white_frame = 255 * np.ones((180, 320, 3), np.uint8)
        # The sudden motion contains a large motion with 80x120 pixels.
        self.sudden_motion_frame = self.black_frame.copy()
        self.sudden_motion_frame[
            20:100,
            80:200,
        ] = 255
        self.sudden_motion_frame[
            0,
            0,
        ] = 255
        self.sudden_motion_frame[
            179,
            319,
        ] = 255


def cap_video(video):
    cap = cv2.VideoCapture(video)
    video_meta = {}
    video_meta["count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_meta["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_meta["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_meta["fps"] = int(cap.get(cv2.CAP_PROP_FPS))

    return cap, video_meta


if __name__ == "__main__":
    path = ""
    video_list = os.listdir(path)

    def split_video_filename(filename: str):
        return {"jumbo_id": filename.split("_")[0], "filename": filename}

    video_list = list(map(split_video_filename, video_list))
    total_video = len(video_list)

    for count, video in enumerate(video_list):
        print(f"{video['filename']}: {int((count+1)/total_video*100)} %", end="\r")

        if not os.path.exists(f'data/{video["jumbo_id"]}'):
            os.makedirs(f'data/{video["jumbo_id"]}', 0o777)

        current_capture, metadata = cap_video(f'{path}{video["filename"]}')
        amf = AdvancedMotionFilter(
            ssc_model="model/scene_knn_model",
            frame_width=metadata["width"],
            frame_height=metadata["height"],
        )

        i = 0
        ret_bool = True
        while i < metadata["count"] and ret_bool:
            ret_bool, frame = current_capture.read()
            resized_frame = video_utils.frame_resize(frame.copy())
            mask = amf.apply(resized_frame)
            Image.fromarray(mask.copy()).save(f'data/{video["jumbo_id"]}/{str(i)}.jpg')
            i += 1
        current_capture.release()
