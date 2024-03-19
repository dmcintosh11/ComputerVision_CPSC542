import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torch.autograd import Variable
import torch

from lib.data_loader import *
from lib.data_transformer import *
from lib.model_utils import *
from lib.model_evaluation import *


#Set random seed
np.random.seed(7)
random.seed(7)
    
mod_name = 'UNetPP'    

#Set directory for saving plots
plot_dir=f'plots/evaluation/{mod_name}/'

models_dir = f'saved_models/{mod_name}/'

# Set the paths to the image and mask folders
image_dir = 'dataset/Images/'
mask_dir = 'dataset/Masks_Binary/'

# Load image and mask paths
image_paths, mask_paths = load_image_mask_paths(image_dir, mask_dir)
mask_paths = load_mask_paths(mask_dir)

#Split data into training, testing and validation sets
# 70% training, 15% testing, 15% validation
train_image_paths, test_image_paths, val_image_paths, train_mask_paths, test_mask_paths, val_mask_paths = split_data(image_paths, mask_paths)

# Instantiate the custom transform
training_transform = CustomTransform(augment=True)
inference_transform = CustomTransform(augment=False)


# Initialize the dataset with augmentation for training
train_dataset = WaterBodiesDataset(train_image_paths, train_mask_paths, transform=training_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

val_dataset = WaterBodiesDataset(val_image_paths, val_mask_paths, transform=inference_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)




# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print()
print(device)
print()

# Initialize a pre-trained U-Net model
model = smp.UnetPlusPlus(
    encoder_name="resnet50",
    encoder_weights="imagenet",  # Use weights pre-trained on ImageNet.
    in_channels=3,  # Number of input channels.
    classes=1,  # Number of output classes.
).to(device)

print('Encoder')
print(model.encoder)
print()
print('Decoder')
print(model.decoder)
print()
from torchsummary import summary

input_size = (3, 224, 224)
summary(model, input_size)

# Define loss function and optimizer
criterion = torch.nn.BCEWithLogitsLoss()  # Appropriate for binary segmentation
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


early_stopping = EarlyStopping(patience=5, verbose=True, save_mod=False)


# Initialize lists to store the training history
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_ious = []
val_ious = []

num_epochs = 25  # Define the number of epochs to train for

best_epoch = 0
best_val_loss = float('inf')
best_val_iou = 0.0

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    running_iou = 0.0
    running_accuracy = 0.0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Calculate IoU
        iou = calculate_iou(outputs, masks)
        running_iou += iou.item()

        # Calculate accuracy
        preds = torch.sigmoid(outputs) > 0.5
        correct = (preds == masks).float()
        accuracy = correct.sum() / correct.numel()
        running_accuracy += accuracy.item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
    train_len = len(train_loader)
    
    train_iou = running_iou / train_len
    train_loss = running_loss / train_len
    train_accuracy = running_accuracy / train_len
    
        
    val_loss, val_iou, val_accuracy = validate(model, val_loader, criterion, device)
    
    # Save the loss, accuracy, and IoU values
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    train_ious.append(train_iou)
    val_ious.append(val_iou)
    
    # Check if this epoch had the lowest validation loss so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        print(f'New best model found!')
        
    if val_iou > best_val_iou:
        best_val_iou = val_iou
    

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Train Accuracy: {train_accuracy:.4f}')
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, Val Accuracy: {val_accuracy:.4f}')
    
    print('Saving model...')
    torch.save(model.state_dict(), f'{models_dir}water_segmentation_model_EPOCH_{epoch+1}.pth')
    
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break
    
    
    print()

    
print(f'Best epoch: {best_epoch+1} with validation loss: {best_val_loss:.4f} and validation IoU: {best_val_iou:.4f}')

# Save the training history
history = {
    'train_loss': train_losses,
    'val_loss': val_losses,
    'train_accuracy': train_accuracies,
    'val_accuracy': val_accuracies,
}
torch.save(history, f'{models_dir}history.pth')


# Plot the training and validation losses
plot_history(models_dir, plot_dir, mod_name)