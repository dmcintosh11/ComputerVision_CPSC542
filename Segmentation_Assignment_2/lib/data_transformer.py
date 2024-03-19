#Library of classes to augment and preprocess data

from torchvision import transforms
import torchvision.transforms.functional as TF
import torch
import random
import numpy as np
from PIL import Image

#Used to apply augmentations and preprocessing data
class CustomTransform:
    def __init__(self, augment=False, image_net_norm=True):
        self.augment = augment
        # Spatial transformations applicable to both images and masks
        self.spatial_augmentation = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=None),
            transforms.RandomHorizontalFlip(),
        ])
        
        # Color transformations for images only
        self.color_augmentation = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ], p=0.5),
        ])

        # Common preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        # Normalization for images in imagenet
        if image_net_norm:
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            self.normalize = lambda x: x

    def __call__(self, image, mask):
        if self.augment:
            # Apply spatial augmentations with the same seed for consistency
            seed = random.randint(0, 10000)
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.spatial_augmentation(image)
            random.seed(seed)
            torch.manual_seed(seed)
            mask = self.spatial_augmentation(mask)

            # Apply color augmentation to image only
            image = self.color_augmentation(image)

        # Apply common preprocessing
        image = self.preprocess(image)
        mask = self.preprocess(mask)
        
        # Normalize image only
        image = self.normalize(image)

        return image, mask



#Wrapper class to format data properly for pytorch
class WaterBodiesDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform  # This should be an instance of CustomTransform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')  # Convert mask to grayscale

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask