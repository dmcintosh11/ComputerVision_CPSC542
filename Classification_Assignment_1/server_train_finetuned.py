#This file handles loading and fine tuning the feature extraction model



# Imports
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import pickle

import data_loaders


#Loads dataset
data_handler = data_loaders.DataHandler('./dataset')

# 70/10/20 train/val/test split
train_df, val_df, test_df = data_handler.get_split_data()
train_gen, valid_gen, test_gen = data_handler.get_generators()

#Sets a smaller learning rate to fine tune on
new_learning_rate = 0.001 / 10

#Loads feature extraction model
model = load_model('./saved_models/DL_models/Weather_CNN_Initial_BESTVALLOSS.h5')


#Freezes top 50 non-BatchNormalization layers
for layer in model.layers[-50:]:
    if not isinstance(layer, BatchNormalization):  # Keep BatchNormalization layers frozen
        layer.trainable = True
        

#Compile model
model.compile(loss = 'categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=new_learning_rate), metrics=['accuracy'])


#Create callbacks
callback  = [EarlyStopping(monitor='val_loss',
                            min_delta=0,
                            patience=4,
                            mode='auto'),
            ModelCheckpoint(
                            './saved_models/DL_models/Weather_CNN_Finetuned_best_BESTVALLOSS.h5',  # Path where to save the model
                            monitor='val_loss',  # The metric to monitor
                            save_best_only=True,  # Only save the best model
                            mode='min',  # The direction of optimization (minimize or maximize the monitored metric)
                            verbose=1  # Log a message when a better model is saved
            )]

#Train model
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=100,
    callbacks=callback
)


#Save model weights
model.save('./saved_models/DL_models/Weather_CNN_Finetuned_best.h5')


#Save history of model training
with open('./saved_models/DL_models/Weather_CNN_Finetuned_best_History.pkl', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


