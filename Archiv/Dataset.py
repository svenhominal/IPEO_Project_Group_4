import os
import shutil
from typing import List, Tuple
import json
import rasterio


class LargeRocksDataset:
    def __init__(self, image_folder: str, json_dataset: str, output_path: str, split = [80,10,10], combined_rgb_hillshade: bool = False):
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
        self.combined_rgb_hillshade = combined_rgb_hillshade
        self.train_percent, self.val_percent, self.test_percent = split
        
    
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
        # Create directories for each split
        for split in self.splits:
            os.makedirs(os.path.join(self.image_dir, split), exist_ok=True)
            os.makedirs(os.path.join(self.label_dir, split), exist_ok=True)
            
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
            
            # Adjust the split based on the provided percentages
            if split == 'test':
                test_index = hash(file_name) % 100
                if test_index < ((self.val_percent*9.8)/3.40):
                    split = 'val'
                elif test_index < ((self.val_percent*9.8)/3.40 + (self.test_percent*9.8)/3.40):
                    split = 'test'
                else:
                    split = 'train'
            
            dst_img_path = os.path.join(self.image_dir, split, file_name)

            if self.combined_rgb_hillshade:
                with rasterio.open(img_path) as src:
                    profile = src.profile
                    profile.update(count=3)  # Update profile to have 3 bands
                    bands = [src.read(1), src.read(4), src.read(3)]  # Replace 2nd band with 4th band (hillshade)
                
                # Save the modified image to the appropriate YOLO image folder
                with rasterio.open(dst_img_path, 'w', **profile) as dst:
                    for i, band in enumerate(bands, start=1):
                        dst.write(band, i)
            else:
                # Copy the image to the appropriate YOLO image folder
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
        print("----" * 10)

    def remove_duplicates_in_labels(self):
        """
        Traverse the labels directory and remove duplicate lines in each label file.
        Print a message only if duplicates were removed.
        """

        #splits = ['labels/val']  # You can expand this list to other splits if needed.
        splits = ['labels/val', 'labels/train', 'labels/test']  # You can expand this list to other splits if needed.
        for split in splits:
            labels_path = os.path.join(self.output_path, split)
            print(labels_path)
            if not os.path.exists(labels_path):
                print(f"Directory not found: {labels_path}")
                continue
            
            for label_file in os.listdir(labels_path):
                file_path = os.path.join(labels_path, label_file)
                
                if not label_file.endswith('.txt'):
                    continue  # Skip non-label files
                
                try:
                    # Read file and remove duplicates
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    # Strip whitespace and remove duplicates while maintaining the original order
                    seen = set()
                    unique_lines = []
                    for line in lines:
                        stripped_line = line.strip()  # Remove leading/trailing spaces and newlines
                        if stripped_line and stripped_line not in seen:  # Check for duplicate lines
                            unique_lines.append(line)  # Append the original line (with all whitespaces)
                            seen.add(stripped_line)
                    
                    # Check if duplicates were removed
                    if len(lines) != len(unique_lines):
                        # Write back the unique lines
                        with open(file_path, 'w') as f:
                            f.writelines(unique_lines)  # Writing back without sorting to preserve original order
                        
                        print(f"Duplicates removed in file: {file_path}")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
        
        print("----" * 10)

    def check_images_and_label_size(self):
        """
        Check if the number of images in the image folder matches the number of annotation entries in the JSON file.
        Also ensures that there is no mismatch between image files and annotations.
        """
        # Load the JSON data
        with open(self.label_file, 'r') as f:
            data = json.load(f)

        # Extract image filenames from the JSON data
        image_files_from_json = [item['file_name'] for item in data['dataset']]

        # List all the image files in the specified folder
        image_files_in_folder = os.listdir(self.image_folder)

        # Filter out non-image files (e.g., directories) if needed
        image_files_in_folder = [f for f in image_files_in_folder if f.endswith('.tif')]

        # Check if the number of images matches the number of entries in the JSON
        num_images_in_json = len(image_files_from_json)
        num_images_in_folder = len(image_files_in_folder)

        # Compare the counts
        if num_images_in_json == num_images_in_folder:
            print(f"Success! The number of images ({num_images_in_folder}) matches the number of annotations ({num_images_in_json}).")

        else:
            print(f"Warning! The number of images ({num_images_in_folder}) does not match the number of annotations ({num_images_in_json}).")
            
            # Optionally, check for missing or extra images
            missing_images = set(image_files_from_json) - set(image_files_in_folder)
            extra_images = set(image_files_in_folder) - set(image_files_from_json)
            
            if missing_images:
                print("Missing images: ", missing_images)
            if extra_images:
                print("Extra images in the folder: ", extra_images)
        
        print("----" * 10)

    def print_actual_split(self):

        train_image_dir = os.path.join(self.output_path, "images/train")
        val_image_dir = os.path.join(self.output_path, "images/test")
        test_image_dir = os.path.join(self.output_path, "images/val")


        train_count = len(os.listdir(train_image_dir))
        val_count = len(os.listdir(val_image_dir))
        test_count = len(os.listdir(test_image_dir))
        total = train_count + val_count + test_count

        print(f"*** {self.output_path} Dataset Split ***")
        print(f"Train set: {train_count} images - {train_count /total * 100:.2f}%")
        print(f"Validation set: {val_count} images - {val_count / total * 100:.2f}%")
        print(f"Test set: {test_count} images - {test_count / total * 100:.2f}%")
        print("----" * 10)