#This file handles loading ResNet50 model and adding feature extraction layers to adapt to weather classification


#Importing Libraries
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
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import pickle

import data_loaders


#Loads dataset
data_handler = data_loaders.DataHandler('./dataset')

# 70/10/20 train/val/test split
train_df, val_df, test_df = data_handler.get_split_data()
train_gen, valid_gen, test_gen = data_handler.get_generators()

#Loads the pre trained model and removes top layer, adds average pooling layer
pre_model = ResNet50(input_shape=(224,224, 3),
                    include_top=False,
                    weights='imagenet',
                    pooling='avg')


#Freezes pre-trained model
pre_model.trainable = False


#Adds two dense layers with relu and 100 neurons along with dropout for regularization
inputs = pre_model.input
x = Dense(100, activation='relu')(pre_model.output)
x = Dropout(0.5)(x)  # Dropout 50% of the neurons
x = Dense(100, activation='relu')(x)
x = Dropout(0.5)(x)  # Dropout 50% of the neurons
outputs = Dense(11, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)


#Compiles model
model.compile(loss = 'categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])


#Sets early stopping and saves the best model (by validation loss)
callback  = [EarlyStopping(monitor='val_loss',
                            min_delta=0,
                            patience=5,
                            mode='auto'),
            ModelCheckpoint(
                            './saved_models/DL_models/Weather_CNN_Initial_BESTVALLOSS.h5',  # Path where to save the model
                            monitor='val_loss',  # The metric to monitor
                            save_best_only=True,  # Only save the best model
                            mode='min',  # The direction of optimization (minimize or maximize the monitored metric)
                            verbose=1  # Log a message when a better model is saved
                            )]
    
#Trains model
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=100,
    callbacks=callback
)

#Saves model
model.save('./saved_models/DL_models/Weather_CNN_Initial.h5')


#Saves history
with open('./saved_models/DL_models/Weather_CNN_Initial_History.pkl', 'wb') as hist_file:
    pickle.dump(history.history, hist_file)



