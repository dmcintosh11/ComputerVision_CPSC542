#This class handles evaluating performance of a deep learning model

#Imports libraries
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from IPython.display import display

pd.options.display.float_format = "{:,.2f}".format



class ModelEvaluator:
    
    #train_gen.class_indices
    def __init__(self, model, history, data_gen, data_df, train_class_indices):
        
        self.model = model
        self.history = history
        self.data_gen = data_gen
        self.data_df = data_df
        self.train_class_indices=train_class_indices
        
        self.make_pred()
        
    #Stores predictions
    def make_pred(self):
        # Predict Data Test
        pred = self.model.predict(self.data_gen)
        self.pred_probs = pred
        pred = np.argmax(pred,axis=1)
        labels = self.train_class_indices
        labels = dict((v,k) for k,v in labels.items())
        pred = [labels[k] for k in pred]
        
        self.pred = pred
        
        return
        
        
    #Prints all evaluation metrics
    def full_report(self, width: int=10, height: int=8) -> None:
        
        self.plot_history()
        
        
        self.plot_examples(width, height)
        
        self.print_report()

        self.plot_cf(width, height)
            
        return
    
    #Plots training history
    def plot_history(self):
        
        # Plotting Accuracy, val_accuracy, loss, val_loss
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        ax = ax.ravel()

        for i, met in enumerate(['accuracy', 'loss']):
            ax[i].plot(self.history[met])
            ax[i].plot(self.history['val_' + met])
            ax[i].set_title('Model {}'.format(met))
            ax[i].set_xlabel('epochs')
            ax[i].set_ylabel(met)
            ax[i].legend(['Train', 'Validation'])
            
        return
    
    
    #Plots examples of model predictions with images        
    def plot_examples(self, width: int=10, height: int=8):
        
        plt.figure(figsize=(width, height))
        
        # Display 16 pictures of the dataset with their labels
        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 8),
                                subplot_kw={'xticks': [], 'yticks': []}, facecolor='w')

        for i, ax in enumerate(axes.flat):
            img = plt.imread(self.data_df.File_Path.iloc[i])
            true_label = self.data_df.Labels.iloc[i]
            pred_label = self.pred[i]
            ax.imshow(img)
            title_color = 'red' if true_label != pred_label else 'black'
            ax.set_title(f"True: {true_label}\nPredicted: {pred_label}", color=title_color)
        plt.tight_layout()

        # Add an overall title to the figure
        fig.suptitle('16 Examples of Images', fontsize=16, color='black')

        # Adjust layout to make room for the figure's title without overlapping subplots
        plt.subplots_adjust(top=0.9)

        plt.show()
        
        return
    
    #Determines if value is below threshold
    @staticmethod
    def __highlight_low_values(val, threshold=0.9):
        """
        Highlights the values in a DataFrame that are below the threshold.
        """
        numeric_val = float(val)
        color = 'red' if numeric_val < threshold else 'white'
        return f'color: {color}'
    
    #Prints  classification report
    def print_report(self) -> None:
        
        print("Classification Report:")
        self.__styled_classification_report(self.data_df.Labels, self.pred)


    #Handles styling the classification report to highlight all subpar values
    def __styled_classification_report(self, y_true, y_pred, threshold=0.9):
        """
        Transforms the classification report into a styled DataFrame
        that highlights concerning values.
        """
        report = classification_report(y_true, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        df_report['support'] = df_report['support'].astype(int)

        # Format numbers as strings with two decimal places, removing trailing zeros
        df_report['precision'] = df_report['precision'].apply(lambda x: f"{x:.2f}")
        df_report['recall'] = df_report['recall'].apply(lambda x: f"{x:.2f}")
        df_report['f1-score'] = df_report['f1-score'].apply(lambda x: f"{x:.2f}")

        # Apply styling
        styled_report = df_report.style.applymap(lambda x: self.__highlight_low_values(x, threshold), subset=['precision', 'recall', 'f1-score'])
        
        # Display the styled DataFrame
        display(styled_report)
        
        return
    
    #Plots confusion matrix
    def plot_cf(self, width: int=10, height: int=8) -> None:
        
        # Calculate the confusion matrix
        conf_mat = confusion_matrix(self.data_df.Labels, self.pred)

        # Normalize the confusion matrix
        conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

        # Increase the size of the figure
        plt.figure(figsize=(width, height))

        # Create the heatmap with annotations and larger font size for clarity
        sns.heatmap(conf_mat_normalized, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=self.train_class_indices, yticklabels=self.train_class_indices,
                    cbar_kws={'label': 'Proportion'})

        # Add labels and title with larger font size
        plt.ylabel('Actual', fontsize=14)
        plt.xlabel('Predicted', fontsize=14)
        plt.title('Normalized Confusion Matrix', fontsize=16)

        # Rotate the tick labels for better visibility if needed
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)

        # Adjust the layout and show the plot
        plt.tight_layout()
        plt.show()
        
        return
    
    
    #Displays the images the model's loss is the highest on
    def display_worst_performing_images(self, num_images: int=5, width: int=10, height: int=8):
        # Calculate the grid size
        grid_size = int(np.ceil(np.sqrt(num_images)))
        
        # Ensure num_images does not exceed the dataset size
        num_images = min(num_images, len(self.data_df))
        
        # Convert true labels to one-hot encoding
        true_labels = pd.get_dummies(self.data_df.Labels).values

        
        # Compute categorical cross-entropy loss for each image
        losses = tf.keras.losses.categorical_crossentropy(true_labels, self.pred_probs)
        
        # Convert tensor to numpy array for manipulation
        losses = losses.numpy()
        
        # Get indices of images with the highest loss
        worst_indices = np.argsort(losses)[-num_images:]
        
        plt.figure(figsize=(width, height))
        
        # Plot the worst performing images in a grid
        fig, axes = plt.subplots(nrows=grid_size, ncols=grid_size, figsize=(grid_size * 4, grid_size * 4),
                                subplot_kw={'xticks': [], 'yticks': []}, facecolor='w')
        
        for i, ax in enumerate(axes.flat):
            if i < num_images:
                # Find the corresponding index in the dataset
                idx = worst_indices[i]
                
                # Load and display the image
                img_path = self.data_df.File_Path.iloc[idx]
                img = plt.imread(img_path)
                ax.imshow(img)
                
                # Get the predicted label with highest probability
                pred_label = self.pred[idx]
                true_label = self.data_df.Labels.iloc[idx]
                
                # Set the title with true and predicted labels
                title = f"True: {true_label}\nPred: {pred_label}\nLoss: {losses[idx]:.2f}"
                ax.set_title(title)
            else:
                ax.axis('off')  # Hide unused subplots
        
        plt.tight_layout()
        
        # Add an overall title to the figure
        fig.suptitle('Top Worst Performing Images', fontsize=16, color='black')
        
        # Adjust layout to make room for the figure's title without overlapping subplots
        plt.subplots_adjust(top=0.9)
        
        plt.show()

        return