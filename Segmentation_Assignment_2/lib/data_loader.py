#Library of functions to load data

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split


def load_image_mask_paths(image_dir, mask_dir):
    image_paths = []
    mask_paths = []

    for file_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, file_name)
        mask_path = os.path.join(mask_dir, file_name)
        image_paths.append(image_path)
        mask_paths.append(mask_path)
        
        # Ensure image_paths and mask_paths are sorted in the same order
        image_paths = sorted(image_paths)
        mask_paths = sorted(mask_paths)

    return image_paths, mask_paths

def load_mask_paths(mask_dir):
    mask_paths = []

    for file_name in os.listdir(mask_dir):
        mask_path = os.path.join(mask_dir, file_name)
        mask_paths.append(mask_path)

        # Ensure image_paths and mask_paths are sorted in the same order
        mask_paths = sorted(mask_paths)

    return mask_paths

#Split the data into training, testing and validation sets
def split_data(image_paths, mask_paths, train_size=0.7, test_size=0.15, val_size=0.15, shuffle=True, random_state=7):
    """
    Splits the image and mask paths into training, testing, and validation sets.

    Parameters:
    - image_paths: A list of paths to the sample images.
    - mask_paths: A list of paths to the corresponding masks of the sample images.
    - train_size: Proportion of the dataset to include in the training set (default is 0.7).
    - test_size: Proportion of the dataset to include in the testing set (default is 0.15).
    - val_size: Proportion of the dataset to include in the validation set (default is 0.15).
    - shuffle: Whether or not to shuffle the data before splitting (default is True).

    Returns:
    - train_image_paths: A list of paths to the training set images.
    - test_image_paths: A list of paths to the testing set images.
    - val_image_paths: A list of paths to the validation set images.
    - train_mask_paths: A list of paths to the training set masks.
    - test_mask_paths: A list of paths to the testing set masks.
    - val_mask_paths: A list of paths to the validation set masks.
    """
    # Check if the proportions add up to 1
    assert train_size + test_size + val_size == 1, "The proportions of train_size, test_size, and val_size should add up to 1."
    
    # Split the data using train_test_split function from sklearn
    train_image_paths, test_image_paths, train_mask_paths, test_mask_paths = train_test_split(image_paths, mask_paths, train_size=train_size, test_size=test_size, shuffle=shuffle, random_state=random_state)
    train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(train_image_paths, train_mask_paths, train_size=(train_size / (train_size + val_size)), test_size=(val_size / (train_size + val_size)), shuffle=shuffle, random_state=random_state)
    
    return train_image_paths, test_image_paths, val_image_paths, train_mask_paths, test_mask_paths, val_mask_paths