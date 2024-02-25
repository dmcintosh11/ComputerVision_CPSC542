#This class handles loading data into a format that is usable to models as well as exploratory
#data analysis functions relating to determining necessity of deep learning



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
from tensorflow.keras.applications.resnet50 import preprocess_input
from IPython.display import display


class DataHandler:
    
    def __init__(self, path: str, random_state: int=7) -> None:
        self.path = path
        self.random_state = random_state
        
        self.__load_data()
        self.__split_data()
        self.__create_generators()
        
        #Grabs class names to label confusion matrix properly
        self.class_names = sorted(self.data['Labels'].unique())


    #Loads and shuffles dataset
    def __load_data(self) -> pd.DataFrame:
        #Loads all path files to images
        path_imgs = list(glob.glob(self.path+'/**/*.jpg'))

        #Grabs the label of every image path
        labels = list(map(lambda x:os.path.split(os.path.split(x)[0])[1], path_imgs))
        file_path = pd.Series(path_imgs, name='File_Path').astype(str)
        labels = pd.Series(labels, name='Labels')

        #Combines the two columns
        data = pd.concat([file_path, labels], axis=1)

        #Shuffles the whole dataset
        data = data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        self.data = data
        
        return


    #Plots all of the information about the data
    def plot_data(self, width: int=15, height: int=7) -> None:
        
        self.class_info()

        self.show_examples(width, height)
        
        self.show_distribution(width, height)
        
        return
    
    #Prints class information
    def class_info(self) -> None:
        
        print(f'Number of classes: {len(pd.unique(self.data["Labels"]))}')
        print(f'All class labels: {pd.unique(self.data["Labels"])}')
        
        return
    
    #Shows examples of images and labels in the data
    def show_examples(self, width: int=15, height: int=7) -> None:

        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(width, height),
                                subplot_kw={'xticks': [], 'yticks': []}, facecolor='w')
        for i, ax in enumerate(axes.flat):
            ax.imshow(plt.imread(self.data.File_Path[i]))
            ax.set_title(self.data.Labels[i])
        plt.tight_layout()

        #Add an overall title to the figure
        fig.suptitle('16 Examples of Images', fontsize=16, color='black')

        #Adjust layout to make room for the figure's title without overlapping subplots
        plt.subplots_adjust(top=0.9)

        plt.show()
        
        return
    
    #Plots the distribution of the classes
    def show_distribution(self, width: int=15, height: int=7) -> None:
        
        counts = self.data.Labels.value_counts()
        plt.figure(figsize=(width, height))
        sns.barplot(x=counts.index, y=counts)
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.xticks(rotation=50)
        plt.title('Distribution of Classes')
        
        return

    #Splits data into 70/10/20 train/val/test split
    def __split_data(self) -> None:
        self.train_df, self.test_df = train_test_split(self.data, test_size=0.2, stratify=self.data['Labels'], random_state=self.random_state)
        self.train_df, self.val_df = train_test_split(self.train_df, test_size=0.125, stratify=self.train_df['Labels'], random_state=self.random_state)  # 0.125 x 0.8 = 0.1

        return
    
    
    #Loads images to be usable by random forest as exploratory model
    @staticmethod
    def load_and_preprocess_image_RF(img_path, img_size: list=[100, 100], use_grayscale: bool=True) -> None:
        image = tf.io.read_file(img_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, img_size)
        
        if use_grayscale:
            image = tf.image.rgb_to_grayscale(image)  # Convert to grayscale
            
        image /= 255.0  # Normalize
        
        return image.numpy().flatten()  # Flatten and convert to numpy array
    
    #Formats data to be input into training a random forest model
    def preprocess_RF(self, img_size: list=[100, 100], use_grayscale: bool=True) -> tuple:
        
        X_train_rf = np.array([self.load_and_preprocess_image_RF(img_path=path, img_size=img_size, use_grayscale=use_grayscale) for path in self.train_df['File_Path']])
        X_val_rf = np.array([self.load_and_preprocess_image_RF(img_path=path, img_size=img_size, use_grayscale=use_grayscale) for path in self.val_df['File_Path']])
        X_test_rf = np.array([self.load_and_preprocess_image_RF(img_path=path, img_size=img_size, use_grayscale=use_grayscale) for path in self.test_df['File_Path']])
        
        return X_train_rf, X_val_rf, X_test_rf
    


    #Highlights values in metrics that are subpar
    @staticmethod
    def __highlight_low_values(val, threshold=0.9):
        """
        Highlights the values in a DataFrame that are below the threshold.
        """
        numeric_val = float(val)
        color = 'red' if numeric_val < threshold else 'black'
        return f'color: {color}'
    
    #Prints classification report with highlights
    def print_report(self, y, predictions) -> None:
        
        print("Classification Report:")
        self.__styled_classification_report(y, predictions)


    #Handles highlighting subpar classification metrics
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


    #Shows classification report and confusion matrix of model
    def evaluate_RF(self, model: object, X, y, show_report: bool=True, show_cf: bool=True) -> None:
        
        #Makes predictions
        predictions = model.predict(X)
        
        #Prints classification report
        if show_report:
            self.print_report(y, predictions)

        #Plots confusion matrix
        if show_cf:
            conf_mat = confusion_matrix(y, predictions)
            sns.heatmap(conf_mat, fmt="d", cmap="Blues", xticklabels=self.class_names, yticklabels=self.class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.show()
        
        return
    
    
    #Creates data generators to be used as neural network inputs with data augmentation
    def __create_generators(self, preprocess_func=preprocess_input, image_size: tuple=(224, 224)):

        datagen = ImageDataGenerator(preprocessing_function=preprocess_func)
            
        self.train_gen = datagen.flow_from_dataframe(
            dataframe=self.train_df,
            x_col='File_Path',
            y_col='Labels',
            target_size=image_size,
            class_mode='categorical',
            batch_size=32,
            shuffle=False,
            seed=self.random_state,
            rotation_range=30,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest")
        self.valid_gen = datagen.flow_from_dataframe(
            dataframe=self.val_df,
            x_col='File_Path',
            y_col='Labels',
            target_size=image_size,
            class_mode='categorical',
            batch_size=32,
            shuffle=False,
            seed=self.random_state)
        self.test_gen = datagen.flow_from_dataframe(
            dataframe=self.test_df,
            x_col='File_Path',
            y_col='Labels',
            target_size=image_size,
            color_mode='rgb',
            class_mode='categorical',
            batch_size=32,
            shuffle=False,
            seed=self.random_state)
        
        
    #Prints breakdown of data shape
    def print_count_breakdown(self):
        
        print(f'# of images total: {self.data.shape}')
        self.class_info()
        print(self.data['Labels'].value_counts())
    
    #Returns full dataset
    def get_data(self) -> pd.DataFrame:
        
        return self.data
    
    #Returns data generators
    def get_generators(self) -> tuple:
        
        return self.train_gen, self.valid_gen, self.test_gen
        
    #Returns data split in dataframes
    def get_split_data(self):
        
        return self.train_df, self.val_df, self.test_df