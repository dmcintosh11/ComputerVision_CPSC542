#Library of functions to assist in training

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from torch.utils.data import DataLoader
import torch

class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0, path='checkpoint.pth', trace_func=print, save_mod=True):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 3
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pth'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.save_mod = save_mod
    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if self.save_mod:
                self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mod:
                self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def calculate_iou(preds, labels):
    smooth = 1e-6
    preds = torch.sigmoid(preds) > 0.5  # Apply threshold to predictions
    preds = preds.float()  # Ensure float type for calculation
    
    #Convert to bool for bitwise operation
    preds = preds.bool()
    labels = labels.bool()

    intersection = (preds & labels).float().sum((1, 2))  # Compute intersection
    union = (preds | labels).float().sum((1, 2))  # Compute union
    
    iou = (intersection + smooth) / (union + smooth)  # Compute IoU
    return iou.mean()  # Return average IoU

def validate(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    running_iou = 0.0
    running_accuracy = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
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

            running_loss += loss.item()

    val_len = len(val_loader)

    average_iou = running_iou / val_len
    average_loss = running_loss / val_len
    average_accuracy = running_accuracy / val_len

    return average_loss, average_iou, average_accuracy

def dice_loss(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)  # Apply sigmoid to get [0,1] probabilities
    pred = pred.view(-1)  # Flatten prediction
    target = target.view(-1)  # Flatten target
    
    intersection = (pred * target).sum()  # Calculate intersection
    dice_coeff = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)  # Calculate Dice coefficient
    
    return 1 - dice_coeff  # Return Dice loss

#Combines Dice loss and BCE
class CombinedLoss(torch.nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        # No need to instantiate dice_loss as it's a standalone function
    
    def forward(self, input, target):
        bce = self.bce_loss(input, target)
        dice = dice_loss(input, target)  # Use the previously defined dice_loss function
        return bce + dice  # Return the combined loss
