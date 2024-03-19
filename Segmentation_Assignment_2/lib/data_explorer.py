#Library of functions to explore data

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def count_filetypes(image_dir, mask_dir):
    image_jpg_count = 0
    image_png_count = 0
    mask_jpg_count = 0
    mask_png_count = 0

    # Count JPG and PNG files in the image directory
    for file_name in os.listdir(image_dir):
        if file_name.endswith('.jpg'):
            image_jpg_count += 1
        elif file_name.endswith('.png'):
            image_png_count += 1

    # Count JPG and PNG files in the mask directory
    for file_name in os.listdir(mask_dir):
        if file_name.endswith('.jpg'):
            mask_jpg_count += 1
        elif file_name.endswith('.png'):
            mask_png_count += 1

    return image_jpg_count, image_png_count, mask_jpg_count, mask_png_count

def visualize_samples(image_paths, mask_paths, sample_indices, plot_dir, plot_title='8 Sample Images and Masks', save_file_name='example_plots.png'):
    """
    Visualizes 8 randomly selected sample images and their corresponding masks for image segmentation in a balanced layout.

    Parameters:
    - image_paths: A list of paths to the sample images.
    - mask_paths: A list of paths to the corresponding masks of the sample images.
    - sample_indices: A list of indices to select the sample images and masks from.
    - plot_dir: Directory path where the plot image will be saved.
    - plot_title: Title of the plot. Default is '8 Sample Images and Masks'.
    - save_file_name: Name of the file to be saved. Default is 'example_plots.png'.
    """
    # Ensure there are enough samples to select from
    assert len(image_paths) >= 8 and len(mask_paths) >= 8, "Not enough samples to visualize."
    
    # 4 rows and 4 columns (2 columns for each image-mask pair)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))  # Adjusted for a square layout
    
    for i, idx in enumerate(sample_indices):
        row = i // 2  # Integer division determines the row
        col = (i % 2) * 2  # Determine starting column for each pair (0 or 2)
        
        img = Image.open(image_paths[idx])
        mask = Image.open(mask_paths[idx])
        
        # Image
        axes[row, col].imshow(img)
        axes[row, col].set_title(f'Original Image')
        axes[row, col].axis('off')  # Hide axes ticks
        
        # Mask
        axes[row, col + 1].imshow(mask, cmap='gray')
        axes[row, col + 1].set_title(f'Mask Image')
        axes[row, col + 1].axis('off')  # Hide axes ticks

    plt.suptitle(plot_title)  # Set the title for the entire plot
    plt.tight_layout()
    plt.savefig(plot_dir + save_file_name)

def check_image_dimensions(image_paths):
    """
    Checks the dimensions of the images and prints the minimum and maximum dimensions.
    """
    image_dims = []
    for img_path in image_paths:
        img = Image.open(img_path)
        width, height = img.size
        channels = len(img.getbands())
        image_dims.append((width, height, channels))

    image_dims = np.array(image_dims)
    print(f'Image dimensions: min={np.min(image_dims, axis=0)}, max={np.max(image_dims, axis=0)}')


def plot_class_distribution(mask_paths, plot_dir, plot_title='Mask Pixel Value Distributions', save_file_name='mask_pixel_value_distributions.png'):
    """
    Calculates the class distribution for the mask pixels (assuming binary masks) and plots the distribution.

    Parameters:
    - mask_paths: A list of paths to the mask images.
    - plot_dir: Directory path where the plot image will be saved.
    - plot_title: Title of the plot. Default is 'Class Distribution'.
    - save_file_name: Name of the file to be saved. Default is 'class_distribution.png'.
    """
    mask_pixels = []
    for mask_path in mask_paths:
        mask = Image.open(mask_path)
        mask_pixels.append(np.array(mask).ravel())

    mask_pixels = np.concatenate(mask_pixels)
    unique_values, counts = np.unique(mask_pixels, return_counts=True)
    class_distribution = dict(zip(unique_values, counts))
    
    plt.figure(figsize=(8, 6))
    plt.bar(class_distribution.keys(), class_distribution.values())
    plt.title(plot_title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Count')
    plt.show()
    
    plt.savefig(plot_dir + save_file_name)

def plot_pixel_value_distribution(image_paths, plot_dir, plot_title='Pixel Value Distribution', save_file_name='pixel_value_distribution.png'):
    """
    Plots the pixel value distribution for one of the images.

    Parameters:
    - image_paths: A list of paths to the images.
    - plot_dir: Directory path where the plot image will be saved.
    - plot_title: Title of the plot. Default is 'Pixel Value Distribution'.
    - save_file_name: Name of the file to be saved. Default is 'pixel_value_distribution.png'.
    """
    img = Image.open(image_paths[0])
    img_pixels = np.array(img).ravel()
    plt.figure(figsize=(8, 6))
    plt.hist(img_pixels, bins=256, range=(0, 256), color='r', alpha=0.5)
    plt.title(plot_title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()
    
    plt.savefig(plot_dir + save_file_name)

def print_dataset_summary(image_paths, mask_paths):
    """
    Prints a summary of the dataset, including the size of images and the size of the dataset.
    """
    # Get the total number of images and masks
    num_images = len(image_paths)
    num_masks = len(mask_paths)
    
    # Check if the number of images and masks match
    if num_images != num_masks:
        print(f"Warning: The number of images ({num_images}) does not match the number of masks ({num_masks}).")
    
    # Get the dimensions of the first image
    first_image = Image.open(image_paths[0])
    width, height = first_image.size
    channels = len(first_image.getbands())
    
    # Assume all images and masks have the same dimensions
    image_size = f"{width} x {height} x {channels}"
    
    # Calculate the total size of the dataset
    total_size = 0
    for img_path in image_paths:
        total_size += os.path.getsize(img_path)
    for mask_path in mask_paths:
        total_size += os.path.getsize(mask_path)
    
    total_size_gb = total_size / (1024 ** 3)  # Convert to GB
    
    print(f"Dataset Summary:")
    print(f"Number of images: {num_images}")
    print(f"Number of masks: {num_masks}")
    print(f"Image size: {image_size}")
    print(f"Total dataset size: {total_size_gb:.2f} GB")