import tifffile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from tqdm import tqdm
import random
import numpy as np

# Paths to the YOLO dataset
image_folder = "YOLO/images/train"
label_folder = "YOLO/labels/train"

def combine_rgb_hillshade(rgb_folder, hillshade_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    rgb_files = sorted(os.listdir(rgb_folder))
    hillshade_files = sorted(os.listdir(hillshade_folder))
    
    for rgb_file, hs_file in tqdm(zip(rgb_files, hillshade_files), total=len(rgb_files)):
        # Read the RGB and hillshade files
        rgb_path = os.path.join(rgb_folder, rgb_file)
        hs_path = os.path.join(hillshade_folder, hs_file)
        
        rgb_image = tifffile.imread(rgb_path)
        hillshade_image = tifffile.imread(hs_path)
        
        # Normalize hillshade to match RGB range (0-255)
        hillshade_image = (hillshade_image / hillshade_image.max() * 255).astype(np.uint8)
        
        # Stack RGB and hillshade to form a 4-channel image
        combined_image = np.dstack((rgb_image, hillshade_image))
        
        # Save the combined image
        output_path = os.path.join(output_folder, rgb_file)
        tifffile.imwrite(output_path, combined_image)

def plot_rgb_hillshade_combinations(rgb_folder, hillshade_folder):
    rgb_files = sorted(os.listdir(rgb_folder))
    hillshade_files = sorted(os.listdir(hillshade_folder))

    # Select 5 random images
    random_indices = random.sample(range(len(rgb_files)), 5)

    fig, axs = plt.subplots(5, 4, figsize=(20, 25))
    
    for idx, random_index in enumerate(random_indices):
        rgb_file = rgb_files[random_index]
        hs_file = hillshade_files[random_index]

        # Read the RGB and hillshade files
        rgb_path = os.path.join(rgb_folder, rgb_file)
        hs_path = os.path.join(hillshade_folder, hs_file)
        
        rgb_image = tifffile.imread(rgb_path)
        hillshade_image = tifffile.imread(hs_path)
        
        # Normalize hillshade to match RGB range (0-255)
        hillshade_image = (hillshade_image / hillshade_image.max() * 255).astype(np.uint8)
        
        # Plot the original RGB image
        axs[idx, 0].imshow(rgb_image)
        axs[idx, 0].axis("off")
        axs[idx, 0].set_title("Original RGB")
        
        # Replace each RGB band with hillshade and plot
        for i in range(3):
            modified_image = rgb_image.copy()
            modified_image[:, :, i] = hillshade_image
            axs[idx, i + 1].imshow(modified_image)
            axs[idx, i + 1].axis("off")
            axs[idx, i + 1].set_title(f"RGB with Hillshade replacing band {i + 1}")
    
    plt.show()

def plot_images_with_comparison(sample_info, label_folder):

    ## Copy of code from usefull_tips.ipynb but with comparison of original and YOLO bounding boxes

    src_SI = 'swissImage_50cm_patches/' + sample_info['file_name']
    src_SS = 'swissSURFACE3D_patches/' + sample_info['file_name']
    src_HS = 'swissSURFACE3D_hillshade_patches/' + sample_info['file_name']
    yolo_label_path = os.path.join(label_folder, sample_info['file_name'].replace('.tif', '.txt'))

    # Read the original bounding boxes
    original_bboxes = sample_info['rocks_annotations']

    # Read YOLO annotations
    yolo_bboxes = []
    if os.path.exists(yolo_label_path):
        with open(yolo_label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            _, x_center, y_center, box_width, box_height = map(float, line.strip().split())
            yolo_bboxes.append((x_center, y_center, box_width, box_height))

    image = tifffile.imread(src_SI)
    dsm = tifffile.imread(src_SS)
    hs = tifffile.imread(src_HS)

    # Plot the images and bounding boxes
    fig, ax = plt.subplots(1, 3, figsize=(15, 15))
    
    # SwissImage
    ax[0].imshow(image)
    ax[0].axis("off")
    ax[0].set_title("swissIMAGE")
    for box in original_bboxes:
        x, y = box['pixel_within_patch_coordinates']
        rect = patches.Circle((x, y), radius=20, linewidth=2, edgecolor='r', facecolor='none')  # Original: red circles
        ax[0].add_patch(rect)
    for xc, yc, bw, bh in yolo_bboxes:
        bw_px = bw * image.shape[1]
        bh_px = bh * image.shape[0]
        xc_px = xc * image.shape[1]
        yc_px = yc * image.shape[0]
        rect = patches.Rectangle((xc_px - bw_px / 2, yc_px - bh_px / 2), bw_px, bh_px, linewidth=2, edgecolor='blue', facecolor='none')  # YOLO: blue rectangles
        ax[0].add_patch(rect)
    
    # DSM
    ax[1].imshow(dsm, cmap="gist_gray")
    ax[1].axis("off")
    ax[1].set_title("swissSURFACE3D")
    for box in original_bboxes:
        x, y = box['pixel_within_patch_coordinates']
        rect = patches.Circle((x, y), radius=20, linewidth=2, edgecolor='r', facecolor='none')  # Original: red circles
        ax[1].add_patch(rect)
    for xc, yc, bw, bh in yolo_bboxes:
        bw_px = bw * dsm.shape[1]
        bh_px = bh * dsm.shape[0]
        xc_px = xc * dsm.shape[1]
        yc_px = yc * dsm.shape[0]
        rect = patches.Rectangle((xc_px - bw_px / 2, yc_px - bh_px / 2), bw_px, bh_px, linewidth=2, edgecolor='blue', facecolor='none')  # YOLO: blue rectangles
        ax[1].add_patch(rect)

    # Hillshade
    ax[2].imshow(hs, cmap="gist_gray")
    ax[2].axis("off")
    ax[2].set_title("Hillshade")
    for box in original_bboxes:
        x, y = box['pixel_within_patch_coordinates']
        rect = patches.Circle((x, y), radius=20, linewidth=2, edgecolor='r', facecolor='none')  # Original: red circles
        ax[2].add_patch(rect)
    for xc, yc, bw, bh in yolo_bboxes:
        bw_px = bw * hs.shape[1]
        bh_px = bh * hs.shape[0]
        xc_px = xc * hs.shape[1]
        yc_px = yc * hs.shape[0]
        rect = patches.Rectangle((xc_px - bw_px / 2, yc_px - bh_px / 2), bw_px, bh_px, linewidth=2, edgecolor='blue', facecolor='none')  # YOLO: blue rectangles
        ax[2].add_patch(rect)
    
    plt.show()

def plot_random_combinations_with_bboxes(rgb_folder, combined_folder, label_folder, yolo_folder):
    rgb_files = sorted(os.listdir(rgb_folder))
    combined_files = sorted(os.listdir(combined_folder))

    # Select 5 random images
    random_indices = random.sample(range(len(rgb_files)), 5)

    fig, axs = plt.subplots(5, 3, figsize=(20, 25))
    
    for idx, random_index in enumerate(random_indices):
        rgb_file = rgb_files[random_index]
        combined_file = combined_files[random_index]

        # Read the RGB and combined files
        rgb_path = os.path.join(rgb_folder, rgb_file)
        combined_path = os.path.join(combined_folder, combined_file)
        
        rgb_image = tifffile.imread(rgb_path)
        combined_image = tifffile.imread(combined_path)
        
        # Read YOLO annotations
        yolo_label_path = os.path.join(label_folder, rgb_file.replace('.tif', '.txt'))
        yolo_bboxes = []
        if os.path.exists(yolo_label_path):
            with open(yolo_label_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                _, x_center, y_center, box_width, box_height = map(float, line.strip().split())
                yolo_bboxes.append((x_center, y_center, box_width, box_height))
        
        # Plot the original RGB image
        axs[idx, 0].imshow(rgb_image)
        axs[idx, 0].axis("off")
        axs[idx, 0].set_title("Original RGB")
        for xc, yc, bw, bh in yolo_bboxes:
            bw_px = bw * rgb_image.shape[1]
            bh_px = bh * rgb_image.shape[0]
            xc_px = xc * rgb_image.shape[1]
            yc_px = yc * rgb_image.shape[0]
            rect = patches.Rectangle((xc_px - bw_px / 2, yc_px - bh_px / 2), bw_px, bh_px, linewidth=2, edgecolor='blue', facecolor='none')  # YOLO: blue rectangles
            axs[idx, 0].add_patch(rect)
        
        # Plot the YOLO image if it exists
        yolo_image_path = os.path.join(yolo_folder, rgb_file)
        if os.path.exists(yolo_image_path):
            yolo_image = tifffile.imread(yolo_image_path)
            axs[idx, 1].imshow(yolo_image)
            axs[idx, 1].axis("off")
            axs[idx, 1].set_title("YOLO Image")
            for xc, yc, bw, bh in yolo_bboxes:
                bw_px = bw * yolo_image.shape[1]
                bh_px = bh * yolo_image.shape[0]
                xc_px = xc * yolo_image.shape[1]
                yc_px = yc * yolo_image.shape[0]
                rect = patches.Rectangle((xc_px - bw_px / 2, yc_px - bh_px / 2), bw_px, bh_px, linewidth=2, edgecolor='blue', facecolor='none')  # YOLO: blue rectangles
                axs[idx, 1].add_patch(rect)
        else:
            axs[idx, 1].axis("off")
            axs[idx, 1].set_title("YOLO Image Not Found (not in train set)")
        
        # Plot the combined RGB and hillshade image
        axs[idx, 2].imshow(combined_image)
        axs[idx, 2].axis("off")
        axs[idx, 2].set_title("Combined RGB and Hillshade")
        for xc, yc, bw, bh in yolo_bboxes:
            bw_px = bw * combined_image.shape[1]
            bh_px = bh * combined_image.shape[0]
            xc_px = xc * combined_image.shape[1]
            yc_px = yc * combined_image.shape[0]
            rect = patches.Rectangle((xc_px - bw_px / 2, yc_px - bh_px / 2), bw_px, bh_px, linewidth=2, edgecolor='blue', facecolor='none')  # YOLO: blue rectangles
            axs[idx, 2].add_patch(rect)
    
    plt.show()

