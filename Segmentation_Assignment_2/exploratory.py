import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

from lib.data_loader import *
from lib.data_explorer import *

def convert_masks_to_binary(source_dir, target_dir, threshold=127):
    """
    Converts all mask JPEG files in the source directory to binary and saves them as PNG files in the target directory.
    
    Parameters:
    - source_dir: Directory containing the original mask JPEG files.
    - target_dir: Directory where the binary PNG mask files will be saved.
    - threshold: Pixel value threshold used for binary conversion (default is 127).
    """
    # Ensure target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Iterate over all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith(".jpg"):  # Check if the file is a JPEG mask
            file_path = os.path.join(source_dir, filename)
            mask = Image.open(file_path)
            mask_array = np.array(mask)
            
            # Apply thresholding
            binary_mask_array = np.where(mask_array > threshold, 255, 0).astype(np.uint8)
            
            # Convert back to an image and save as PNG
            binary_mask = Image.fromarray(binary_mask_array)
            target_file_path = os.path.join(target_dir, os.path.splitext(filename)[0] + ".png")
            binary_mask.save(target_file_path)

    print(f"All masks have been converted and saved to {target_dir}.")
    
#Set random seed
np.random.seed(7)
random.seed(7)
    
#Set directory for saving plots
plot_dir='plots/exploratory/'

# Set the paths to the image and mask folders
image_dir = 'dataset/Images/'
mask_dir = 'dataset/Masks/'

# Count files
image_jpg_count, image_png_count, mask_jpg_count, mask_png_count = count_filetypes(image_dir, mask_dir)
print(f"Image folder: {image_jpg_count} JPG files, {image_png_count} PNG files")
print(f"Mask folder: {mask_jpg_count} JPG files, {mask_png_count} PNG files")

# Load image and mask paths
image_paths, mask_paths = load_image_mask_paths(image_dir, mask_dir)

#Grabs 8 random indices from the image_paths
random_indices = np.random.choice(len(image_paths), 8, replace=False)

# Visualize sample images and masks
visualize_samples(image_paths, mask_paths, random_indices, plot_dir, plot_title='8 Sample Images and Masks Before Mask Correction', save_file_name='example_plots_before_mask_binaries.png')

# Calculate class distribution
plot_class_distribution(mask_paths, plot_dir)

#This distribution showed that the masks are not binary, so we will convert them to binary
    
# Convert masks to binary since they are stored as jpg and the pixel values are not binary
source_dir = 'dataset/Masks/'
target_dir = 'dataset/Masks_Binary/'
#convert_masks_to_binary(source_dir, target_dir)

#Sets the mask directory to the new binary mask directory
mask_dir = target_dir

#Reloads mask paths with the new binary mask directory
mask_paths = load_mask_paths(mask_dir)

# Plot class value distribution of new mask images to confirm binary conversion
plot_class_distribution(mask_paths, plot_dir, plot_title='Binary Mask Class Distribution', save_file_name='binary_mask_pixel_value_distributions.png')

# Print dataset summary
print_dataset_summary(image_paths, mask_paths)

# Visualize sample images and masks
visualize_samples(image_paths, mask_paths, random_indices, plot_dir)