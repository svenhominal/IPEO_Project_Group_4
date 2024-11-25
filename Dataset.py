import os
import shutil
from typing import List, Tuple
import json


class LargeRocksDataset:
    def __init__(self, image_folder: str, json_dataset: str, output_path: str):
        """
        Initialize the dataloader 
        
        Args:
            image_folder (str): Path to folder containing `.tif` images -> for now choose between modalities
            label_file (str): Path to JSON dataset file
            output_path (str): Path to save YOLOv8 formatted dataset -> to keep original dataset intact
        """
        self.image_folder = image_folder  
        self.label_file = json_dataset  
        self.output_path = output_path 
        
        # Define directories for splits
        self.splits = ["train", "val", "test"]
        self.image_dir = os.path.join(output_path, "images")
        self.label_dir = os.path.join(output_path, "labels")
        
        # Create directories for each split
        for split in self.splits:
            os.makedirs(os.path.join(self.image_dir, split), exist_ok=True)
            os.makedirs(os.path.join(self.label_dir, split), exist_ok=True)
    
    def _convert_bbox(self, rel_loc: Tuple[float, float], bbox_size: Tuple[int, int], img_size: Tuple[int, int]) -> List[float]:
        """
        Convert bounding box info to YOLO format: [class_id, x_center, y_center, width, height]. (Only one class for rocks)
        
        Args:
            rel_loc (Tuple[float, float]): Relative location of the object in the image (normalized).
            bbox_size (Tuple[int, int]): Size of the bounding box in pixels.
            img_size (Tuple[int, int]): Image size (width, height).
        
        Returns:
            List[float]: Bounding box in YOLO format.
        """
        x_center, y_center = rel_loc
        width = bbox_size[0] / img_size[0]
        height = bbox_size[1] / img_size[1]
        return [0, x_center, y_center, width, height]  # class_id = 0 for rocks
    
    def process_dataset(self):
        """
        Process the dataset + convert it to YOLOv8 format with train/val/test splits.
        """
        # Load the annotations JSON
        with open(self.label_file, 'r') as f:
            data = json.load(f)
        
        # Iterate over each image in the dataset
        for tile in data['dataset']:
            file_name = tile['file_name']
            img_path = os.path.join(self.image_folder, file_name)
            
            # Check if the image exists
            if not os.path.exists(img_path):
                print(f"Image {img_path} not found. Skipping.")
                continue
            
            img_width, img_height = tile['width'], tile['height']
            annotations = tile.get('rocks_annotations', [])
            split = tile.get('split', "train")  # Default to 'train' if no split is specified
            
            # Copy the image to the appropriate YOLO image folder
            dst_img_path = os.path.join(self.image_dir, split, file_name)
            shutil.copy(img_path, dst_img_path)
            
            # Prepare labels for this image
            label_lines = []
            for annotation in annotations:
                rel_loc = annotation['relative_within_patch_location']
                bbox_size = annotation.get('bbox_size', [30, 30])  #!!! No bbox size in the dataset -> default to 30x30
                yolo_bbox = self._convert_bbox(rel_loc, bbox_size, (img_width, img_height))
                label_lines.append(" ".join(map(str, yolo_bbox)))
            
            # Save labels to the appropriate folder
            label_file = os.path.join(self.label_dir, split, f"{os.path.splitext(file_name)[0]}.txt")
            with open(label_file, 'w') as lf:
                lf.write("\n".join(label_lines))
        
        print(f"Dataset ({self.image_folder}) converted to YOLO format with train/val/test splits at {self.output_path}")
