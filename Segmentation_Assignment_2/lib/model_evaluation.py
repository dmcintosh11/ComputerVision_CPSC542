# Library of functions to evaluate image segmentation models

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import pandas as pd

#Unnormalizes image to display
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)  # reverse the normalization
        return tensor

#Plots history of training
def plot_history(models_dir, plot_dir, mod_name):
    history = torch.load(os.path.join(models_dir, 'history.pth'))
    train_losses = history['train_loss']
    val_losses = history['val_loss']
    train_accuracies = history['train_accuracy']
    val_accuracies = history['val_accuracy']
    num_epochs = len(train_losses)

    # Plotting Loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs), train_losses, label='Training Loss')
    plt.plot(range(num_epochs), val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f'{mod_name}_loss_curve.png'))
    plt.show()

    # Plotting Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs), train_accuracies, label='Training Accuracy')
    plt.plot(range(num_epochs), val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f'{mod_name}_accuracy_curve.png'))
    plt.show()

def calculate_iou(outputs, masks):
    smooth = 1e-6
    outputs = torch.sigmoid(outputs) > 0.5
    outputs = outputs.float()
    masks = masks.float()
    
    intersection = (outputs * masks).sum((1, 2))
    union = outputs.sum((1, 2)) + masks.sum((1, 2)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.mean()

def calculate_confusion_matrix_components(outputs, masks):
    TP = ((outputs == 1) & (masks == 1)).sum().item()
    TN = ((outputs == 0) & (masks == 0)).sum().item()
    FP = ((outputs == 1) & (masks == 0)).sum().item()
    FN = ((outputs == 0) & (masks == 1)).sum().item()
    return TP, TN, FP, FN

def calculate_accuracy(TP, TN, FP, FN):
    total_samples = TP + TN + FP + FN
    correct_predictions = TP + TN
    accuracy = correct_predictions / total_samples
    return accuracy

def calculate_precision(TP, FP):
    return TP / (TP + FP) if TP + FP > 0 else 0

def calculate_recall(TP, FN):
    return TP / (TP + FN) if TP + FN > 0 else 0

def calculate_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

#Main function that calls on necessary functions to give report
def validate_and_visualize(model, data_loader, criterion, device, num_examples, plot_dir, unnormalize=None, use_grad_cam=True, data_name='Test'):
    model.eval()
    running_loss = 0.0
    confusion_matrix = np.zeros((2, 2), dtype=np.int32)
    all_images, all_masks, all_preds, all_losses = [], [], [], []
    total_iou = 0.0
    total_accuracy = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            preds = torch.sigmoid(outputs) > 0.5

            batch_losses = [criterion(output.unsqueeze(0), mask.unsqueeze(0)).item() for output, mask in zip(outputs, masks)]
            all_losses.extend(batch_losses)
            TP, TN, FP, FN = calculate_confusion_matrix_components(preds, masks)
            confusion_matrix += np.array([[TP, FP], [FN, TN]])
            running_loss += loss.item()
            total_iou += calculate_iou(outputs, masks).item() * images.size(0)
            total_accuracy += calculate_accuracy(TP, TN, FP, FN) * images.size(0)
            total_samples += images.size(0)
            if unnormalize:
                images = torch.stack([unnormalize(img) for img in images])
            all_images.extend(images.cpu().detach())
            all_masks.extend(masks.cpu().detach())
            all_preds.extend(preds.cpu().detach())
            
    average_loss = sum(all_losses) / total_samples    
    average_iou = total_iou / total_samples
    average_accuracy = total_accuracy / total_samples
    
    # After the loop collecting predictions and metrics
    TP, FP, FN, TN = confusion_matrix.ravel()

    precision = calculate_precision(TP, FP)
    recall = calculate_recall(TP, FN)
    f1_score = calculate_f1_score(precision, recall)

    metrics_df = pd.DataFrame({
        'Metric': ['Loss', 'IoU', 'Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [average_loss, average_iou, average_accuracy, precision, recall, f1_score]
    })

    print("\nPerformance Metrics:")
    print(metrics_df.to_string(index=False))

    metrics_df.to_csv(os.path.join(plot_dir, f'{data_name}_performance_metrics.csv'), index=False)
    
    
    sorted_indices = np.argsort(all_losses)
    best_indices = sorted_indices[:num_examples]
    worst_indices = sorted_indices[-num_examples:]
    best_images, best_masks, best_preds, best_losses = [all_images[i] for i in best_indices], [all_masks[i] for i in best_indices], [all_preds[i] for i in best_indices], [all_losses[i] for i in best_indices]
    worst_images, worst_masks, worst_preds, worst_losses = [all_images[i] for i in worst_indices], [all_masks[i] for i in worst_indices], [all_preds[i] for i in worst_indices], [all_losses[i] for i in worst_indices]


    

    if use_grad_cam:
        target_layers = [model.encoder.layer4[-1]] 
        
        print('Model Architecture:')
        print(model.encoder)

        # Visualization of examples (best and worst) with Grad-CAM
        visualize_performance_examples_with_grad_cam(model, best_images, best_masks, best_preds, best_losses, num_examples, plot_dir, target_layers, 'best', unnormalize, device)
        visualize_performance_examples_with_grad_cam(model, worst_images, worst_masks, worst_preds, worst_losses, num_examples, plot_dir, target_layers, 'worst', unnormalize, device)
        
    else:
        # Visualization of examples (best and worst)
        visualize_performance_examples(best_images, best_masks, best_preds, best_losses, num_examples, plot_dir, 'best', unnormalize)
        visualize_performance_examples(worst_images, worst_masks, worst_preds, worst_losses, num_examples, plot_dir, 'worst', unnormalize)

    # Plotting the confusion matrix
    class_names = ['Non-Water', 'Water']
    plot_confusion_matrix(confusion_matrix, class_names, plot_dir, f'{data_name.capitalize()}_confusion_matrix.png', plot_title=f'{data_name.capitalize()} Confusion Matrix')

    return metrics_df







def visualize_performance_examples(example_images, example_masks, example_preds, image_losses, n, plot_dir, file_prefix, unnormalize=None):
    """
    Display the top n performing images based on loss, along with their masks and predictions.
    
    Parameters:
    example_images (list of Tensor): The list of image tensors.
    example_masks (list of Tensor): The list of ground truth mask tensors.
    example_preds (list of Tensor): The list of predicted mask tensors.
    image_losses (list of float): The list of losses corresponding to each example.
    n (int): The number of examples to display.
    plot_dir (str): Directory where the plots will be saved.
    file_prefix (str): Prefix for the file name of the saved plot.
    unnormalize (callable, optional): Function to unnormalize images for visualization.
    """
    indices = np.argsort(image_losses)[:n]  # Sort losses and pick top n for visualization

    num_rows = len(indices)
    fig, axs = plt.subplots(num_rows, 3, figsize=(18, 5 * num_rows))

    for i, idx in enumerate(indices):
        image = example_images[idx]
        mask = example_masks[idx]
        pred = example_preds[idx]
        loss = image_losses[idx]

        if unnormalize:
            image = unnormalize(image)

        # Original Image
        img_np = image.cpu().numpy().transpose((1, 2, 0))
        axs[i, 0].imshow(np.clip(img_np, 0, 1))
        axs[i, 0].set_title(f'Original Image (Loss: {loss:.4f}')

        # Ground Truth Mask
        mask_np = mask.cpu().numpy().squeeze()
        axs[i, 1].imshow(mask_np, cmap='gray')
        axs[i, 1].set_title('Ground Truth Mask')

        # Predicted Mask
        pred_np = pred.cpu().numpy().squeeze()
        axs[i, 2].imshow(pred_np, cmap='gray')
        axs[i, 2].set_title('Predicted Mask')

    for ax in axs.ravel():
        ax.axis('off')
        
    fig.suptitle('Performance Examples', fontsize=16, y=1.02)


    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{file_prefix}.png'))
    plt.close()



def plot_confusion_matrix(confusion_matrix, class_names, plot_dir='.', file_name='confusion_matrix.png', normalize=True, plot_title='Confusion Matrix'):
    """
    Plots a confusion matrix, normalized or not.
    
    Parameters:
        confusion_matrix (np.array): The confusion matrix to plot.
        class_names (list of str): List of class names, in the order they index the matrix.
        plot_dir (str): Directory where the plot will be saved.
        file_name (str): File name for the saved plot.
        normalize (bool): Whether to normalize the values in the confusion matrix.
    """
    if normalize:
        # Normalize the confusion matrix.
        cm_sum = np.sum(confusion_matrix, axis=1, keepdims=True)
        confusion_matrix_normalized = confusion_matrix / cm_sum.astype(float) * 100  # Convert to percentage
        fmt = '.2f'  # Format string for annotations inside the heatmap
        annot = np.array([["{0:.2f}%".format(value) for value in row] for row in confusion_matrix_normalized])
    else:
        confusion_matrix_normalized = confusion_matrix
        fmt = 'd'
        annot = np.array([[f"{value}" for value in row] for row in confusion_matrix_normalized])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_matrix_normalized, annot=annot, fmt='', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=normalize)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(plot_title)
    plt.savefig(os.path.join(plot_dir, file_name))
    plt.show()


import pandas as pd

def calculate_precision_recall_f1(TP, TN, FP, FN):
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
    return precision, recall, f1_score

def create_metrics_table(confusion_matrix):
    # Calculate components of confusion matrix
    TP, TN, FP, FN = confusion_matrix[0, 0], confusion_matrix[1, 1], confusion_matrix[0, 1], confusion_matrix[1, 0]
    
    # Calculate Accuracy
    accuracy = calculate_accuracy(TP, TN, FP, FN)
    
    # Calculate IoU
    iou = calculate_iou_from_components(TP, FP, FN)
    
    # Calculate Precision, Recall, F1-Score
    precision, recall, f1_score = calculate_precision_recall_f1(TP, TN, FP, FN)
    
    # Create a dictionary of metrics
    metrics = {
        'Precision': [precision],
        'Recall': [recall],
        'F1-Score': [f1_score],
        'IoU': [iou],
        'Accuracy': [accuracy]
    }
    
    # Convert dictionary to Pandas DataFrame for a nice table format
    metrics_df = pd.DataFrame(metrics)
    
    return metrics_df

def calculate_iou_from_components(TP, FP, FN, smooth=1e-6):
    intersection = TP
    union = TP + FP + FN
    iou = (intersection + smooth) / (union + smooth)
    return iou








def apply_grad_cam(model, input_tensor, target_mask, target_layers, device='cuda'):
    """
    Applies Grad-CAM to given model and input tensor, focusing on specified target class.
    Parameters:
        model: The model to analyze.
        input_tensor: The input tensor for which Grad-CAM is computed.
        target_mask: Binary mask for the target class (1 for target class, 0 otherwise).
        target_layers: Layers of the model to target for Grad-CAM.
        device: The device to use ('cuda' or 'cpu').
    Returns:
        An image tensor with Grad-CAM overlay.
    """
    # Ensure input and model are on the correct device
    input_tensor = input_tensor.to(device)
    model.to(device)
    

    
    # Define the target for Grad-CAM
    targets = [SemanticSegmentationTarget(mask=target_mask)]

    
    cam_image = None

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        cam_image = show_cam_on_image(input_tensor.cpu().numpy().squeeze().transpose(1, 2, 0), grayscale_cam, use_rgb=True)
    
    return cam_image

# For multi class predictions that aren't binary, use the following class instead:
# class SemanticSegmentationTarget:
#     def __init__(self, category, mask):
#         self.category = category
#         self.mask = torch.from_numpy(mask).cuda()
    
#     def __call__(self, model_output):
#         return (model_output[self.category, :, :] * self.mask).sum()

class SemanticSegmentationTarget:
    def __init__(self, mask):
        self.mask = torch.from_numpy(mask).cuda()
    
    def __call__(self, model_output):
        return (model_output[0] * self.mask).sum()





def visualize_performance_examples_with_grad_cam(model, example_images, example_masks, example_preds, image_losses, n, plot_dir, target_layers, file_prefix, unnormalize=None, device='cuda'):
    """
    Display the top n performing images based on loss, along with their masks, predictions, and Grad-CAM visualizations.
    
    Parameters:
    model: The segmentation model being evaluated.
    example_images (list of Tensor): The list of image tensors.
    example_masks (list of Tensor): The list of ground truth mask tensors.
    example_preds (list of Tensor): The list of predicted mask tensors.
    image_losses (list of float): The list of losses corresponding to each example.
    n (int): The number of examples to display.
    plot_dir (str): Directory where the plots will be saved.
    target_layers: Layers of the model to target for Grad-CAM.
    file_prefix (str): Prefix for the file name of the saved plot.
    unnormalize (callable, optional): Function to unnormalize images for visualization.
    device (str): The device to use ('cuda' or 'cpu').
    """
    indices = np.argsort(image_losses)[:n]  # Sort losses and pick top n for visualization

    num_rows = len(indices)
    fig, axs = plt.subplots(num_rows, 4, figsize=(24, 5 * num_rows))

    for i, idx in enumerate(indices):
        image = example_images[idx]
        mask = example_masks[idx]
        pred = example_preds[idx]
        loss = image_losses[idx]

        if unnormalize:
            image = unnormalize(image)

        # Original Image
        img_np = image.cpu().numpy().transpose((1, 2, 0))
        axs[i, 0].imshow(np.clip(img_np, 0, 1))
        axs[i, 0].set_title(f'Original Image (Loss: {loss:.4f})')

        # Ground Truth Mask
        mask_np = mask.cpu().numpy().squeeze()
        axs[i, 1].imshow(mask_np, cmap='gray')
        axs[i, 1].set_title('Ground Truth Mask')

        # Predicted Mask
        pred_np = pred.cpu().numpy().squeeze()
        axs[i, 2].imshow(pred_np, cmap='gray')
        axs[i, 2].set_title('Predicted Mask')

        # Grad-CAM Overlay
        # Prepare the input tensor for Grad-CAM
        input_tensor = example_images[idx].unsqueeze(0)  # Add batch dimension
        target_mask = pred.squeeze().cpu().numpy()  # Use prediction as target mask for Grad-CAM
        cam_image = apply_grad_cam(model, input_tensor, target_mask, target_layers, device)
        axs[i, 3].imshow(cam_image)
        axs[i, 3].set_title('Grad-CAM Overlay')

    for ax in axs.ravel():
        ax.axis('off')
        
    fig.suptitle(f'{file_prefix.capitalize()} Loss Examples', fontsize=20, y=0.995)


    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{file_prefix}_with_grad_cam.png'))
    plt.close()


