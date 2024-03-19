import os
import torch
import matplotlib.pyplot as plt



import numpy as np
from PIL import Image
import random
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torch

from lib.data_loader import *
from lib.data_transformer import *
from lib.model_utils import *
from lib.model_evaluation import plot_history, validate_and_visualize, UnNormalize



#Set random seed
np.random.seed(7)
random.seed(7)

mod_name = 'UNet'

# Set the directory for saved models
models_dir = f'saved_models/{mod_name}'

#Set directory for saving plots
plot_dir=f'plots/evaluation/{mod_name}'



best_epoch = 12


# Set the paths to the image and mask folders
image_dir = 'dataset/Images/'
mask_dir = 'dataset/Masks_Binary/'



# Load image and mask paths
image_paths, mask_paths = load_image_mask_paths(image_dir, mask_dir)
mask_paths = load_mask_paths(mask_dir)

#Split data into training, testing and validation sets
# 70% training, 15% testing, 15% validation
train_image_paths, test_image_paths, val_image_paths, train_mask_paths, test_mask_paths, val_mask_paths = split_data(image_paths, mask_paths)

# Instantiate the custom transform for inference
inference_transform = CustomTransform(augment=False)

train_dataset = WaterBodiesDataset(train_image_paths, train_mask_paths, transform=inference_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)

# Initialize the test dataset
test_dataset = WaterBodiesDataset(test_image_paths, test_mask_paths, transform=inference_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = torch.nn.BCEWithLogitsLoss()  # Appropriate for binary segmentation


# Initialize the best model
best_model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
).to(device)

# Load the best model weights
best_model.load_state_dict(torch.load(os.path.join(models_dir, f'water_segmentation_model_EPOCH_{best_epoch}.pth')))

# Set the model to evaluation mode
best_model.eval()

unnormalize = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
print()

print('Train data:')
train_metrics = validate_and_visualize(best_model, train_loader, criterion, device, 5, f'{plot_dir}/Train/', unnormalize=unnormalize, data_name='Train')

print()

print('Test data:')
test_metrics = validate_and_visualize(best_model, test_loader, criterion, device, 5, f'{plot_dir}/Test/', unnormalize=unnormalize, data_name='Test')


# Merge the train and test metrics DataFrames
combined_metrics = pd.merge(train_metrics, test_metrics, on='Metric')

combined_metrics = combined_metrics.rename(columns={'Value_x': 'Train', 'Value_y': 'Test'})

# Display the combined metrics
print("\nCombined Train and Test Metrics:")
print(combined_metrics.to_string(index=False))

combined_metrics.to_csv(f'{plot_dir}/combined_metrics.csv', index=False)