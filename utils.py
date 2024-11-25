import tifffile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import random

# Paths to the YOLO dataset
image_folder = "YOLO/images/train"
label_folder = "YOLO/labels/train"

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