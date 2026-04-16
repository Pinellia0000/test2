import os
import glob
import csv
from pathlib import Path

import cv2
from tqdm import tqdm
import torch
import numpy as np

# ==========================
# 🔥 终极屏蔽：彻底禁止出错的模块
# ==========================
import sys
sys.modules['scipy.ndimage.interpolation'] = type('fake', (), {'zoom': lambda *args: args[0]})()
sys.modules['scipy'] = type('fake', (), {})()

# 修复 numpy 废弃类型
np.int = int
np.float = float
np.bool = bool
np.complex = complex

import warnings
warnings.filterwarnings("ignore")

torch.serialization.add_safe_globals([type, type(None)])

# 现在可以安全导入 SAN
from SAN.san_api import SanLandmarkDetector

# ==================================================================

def print_directory_structure(root_dir, indent="", directory_name="", is_root=True):
    if is_root:
        print(f"{directory_name}目录结构如下：\n")
    items = sorted(os.listdir(root_dir))
    for idx, item in enumerate(items):
        path = os.path.join(root_dir, item)
        pointer = "└── " if idx == len(items) - 1 else "├── "
        print(indent + pointer + item)
        if os.path.isdir(path):
            extension = "    " if idx == len(items) - 1 else "│   "
            print_directory_structure(path, indent + extension, directory_name, is_root=False)


class LandmarkDetector:
    def __init__(self, model_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.det = SanLandmarkDetector(model_path, device)

    def cal(self, img, offset=None, face_box=None):
        if face_box is None:
            face_box = (0, 0, img.shape[1], img.shape[0])
        locs, _ = self.det.detect(img, face_box)
        x_list = [loc[0] if offset is None else loc[0] - offset[0] for loc in locs]
        y_list = [loc[1] if offset is None else loc[1] - offset[1] for loc in locs]
        return x_list, y_list


def get_img_count(cropped_root_path):
    count = 0
    # 这里修复：必须加 .iterdir()
    for sub_item in Path(cropped_root_path).iterdir():
        if sub_item.is_dir():
            count += len(glob.glob(os.path.join(str(sub_item), "*.jpg")))
    return count


def record_csv(csv_path, rows):
    if os.path.exists(csv_path):
        os.remove(csv_path)
    with open(csv_path, 'w', newline='') as f:
        csv_w = csv.writer(f)
        csv_w.writerows(rows)


def record_face_and_landmarks(cropped_root_path):
    if not os.path.exists(cropped_root_path):
        print(f"path {cropped_root_path} is not exist")
        exit(1)

    sum_count = get_img_count(cropped_root_path)
    print("img count = ", sum_count)

    landmark_model_path = '/kaggle/input/datasets/garlic0000/san-model/checkpoint_49.pth.tar'

    # 强制修复 torch 加载
    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)

    torch.load = patched_load

    landmark_detector = LandmarkDetector(landmark_model_path)

    with tqdm(total=sum_count) as tq:
        for sub_item in Path(cropped_root_path).iterdir():
            if not sub_item.is_dir():
                continue

            img_path_list = glob.glob(os.path.join(str(sub_item), "*.jpg"))
            if len(img_path_list) == 0:
                continue

            img_path_list.sort()
            rows_face = []
            rows_landmark = []

            csv_face_path = os.path.join(str(sub_item), "face.csv")
            csv_landmark_path = os.path.join(str(sub_item), "landmarks.csv")

            for index, img_path in enumerate(img_path_list):
                img = cv2.imread(img_path)
                try:
                    h, w = img.shape[:2]
                    left, top, right, bottom = 0, 0, w, h
                    x_list, y_list = landmark_detector.cal(img, face_box=(left, top, right, bottom))
                except Exception as e:
                    print(f"subject: {sub_item.name}, index: {index}, 错误: {e}")
                    break

                rows_face.append((left, top, right, bottom))
                rows_landmark.append(x_list + y_list)
                tq.update()

            if len(rows_face) == len(img_path_list):
                record_csv(csv_face_path, rows_face)
                record_csv(csv_landmark_path, rows_landmark)


if __name__ == "__main__":
    cropped_root_path = '/kaggle/working/stuTest_retinaface'
    record_face_and_landmarks(cropped_root_path)
    print_directory_structure(cropped_root_path, directory_name='stuTest_retinaface')
