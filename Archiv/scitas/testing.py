## 1. Train / test split

import json
from collections import Counter
import random
from Dataset import LargeRocksDataset
from utils import *
from ultralytics import YOLO

label_file = "large_rock_dataset.json"

with open(label_file, 'r') as f:
    data = json.load(f)

splits = [tile.get('split', 'train') for tile in data['dataset']]  # Default to 'train' if missing
split_counts = Counter(splits)

for split, count in split_counts.items():
    print(f"{split.capitalize()}: {count} images")
    print(f"Percentage: {count / len(splits) * 100:.2f}%")


image_folder = "swissImage_50cm_patches"  # Path to swissImage_50cm_patches or equivalent
label_file = "large_rock_dataset.json"  # JSON file with annotations
output_path = "YOLO"  # Directory to save processed dataset

rocks_dataset = LargeRocksDataset(image_folder, label_file, output_path)
rocks_dataset.process_dataset()

# Paths to the YOLO dataset
image_folder = "YOLO/images/train"
label_folder = "YOLO/labels/train"
json_file_path = 'large_rock_dataset.json'

# Load the JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)

train_data = [tile for tile in data['dataset'] if tile.get('split') == 'train']

model = YOLO("yolov8n.pt")
model.info()

results = model.train(data="data.yml", epochs=100, batch=32, imgsz=640,overlap_mask=False,mask_ratio=0,dropout=0.05,flipud=0.1,shear = 10,degrees=10)
"""
mode=train, model=yolov8n.pt, data=data.yml, epochs=30, time=None, 
patience=100, batch=30, imgsz=640, save=True, save_period=-1, cache=False, 
device=mps, workers=8, project=None, name=train2, exist_ok=False, pretrained=True, 
optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, 
cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, 
freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, 
split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, 
dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, 
agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, 
save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, 
format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, 
workspace=None, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, 
warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, nbs=64, hsv_h=0.015, 
hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, 
bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4,
crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=runs/detect/train2
"""