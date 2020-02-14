# import the necessary packages
import os
import zipfile
import matplotlib
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import sys
import matplotlib.image as mpimg
from glob import glob 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LogNorm

#Keras
import tensorflow as tf
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
from keras import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model
#Sklearn helpers for model evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#cnn_model
from cnn_model import vgg_model

# Hyperparameter dictionary for our project

config = { 'LR' : 0.001,
    'BATCH_SIZE' : 32,
    'NEPOCHS' : 15,
    'IMG_WIDTH' : 75, 
    'IMG_HEIGHT' : 75, 
    'VALIDATION_RATIO' : 0.2}

# train & dev + test directories

base_dir = 'data/'
test_dir = 'test_data/'

# Initialize the Image data generator

datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range = 20,
    fill_mode = 'nearest',
    horizontal_flip=True,
    validation_split= config['VALIDATION_RATIO']) # set validation split

test_datagen = ImageDataGenerator(rescale=1./255)

# Generate the training examples from the base directory using the Image 
# DataGenerator defined above

train_generator = datagen.flow_from_directory(
    base_dir,
    batch_size=config['BATCH_SIZE'],
    target_size=(config['IMG_WIDTH'], config['IMG_HEIGHT']),
    class_mode='categorical',
    subset='training') 

# Generate the training examples from the base directory using the Image 
# DataGenerator defined above

validation_generator = datagen.flow_from_directory(
    base_dir, 
    target_size=(config['IMG_WIDTH'], config['IMG_HEIGHT']),
    batch_size=config['BATCH_SIZE'],
    class_mode='categorical',
    subset='validation') 


# Generate the test examples from the test directory but set shuffle to False so 
# that we can keep track of the labels 

test_generator = test_datagen.flow_from_directory(
                        test_dir,
    			target_size=(config['IMG_WIDTH'], config['IMG_HEIGHT']),
    			batch_size=config['BATCH_SIZE'],
    			class_mode='categorical',
			shuffle = False) 


# Initializing and compiling the model

model = vgg_model(config['IMG_WIDTH'], config['IMG_HEIGHT'], retrain = False)
model.compile(loss = 'categorical_crossentropy', optimizer=Adam(lr = config['LR']), metrics=['accuracy'])

# Set up a learning rate Scheduler

def scheduler(epoch):
  '''
  learning rate scheduler; keeps the learning rate 10^(-3) until 10 and then exponential decrease
  '''
  if epoch < 10:
    return config['LR']
  else:
    return config['LR'] * np.exp(0.1 * (10 - epoch))

learning_rate_scheduler = LearningRateScheduler(scheduler)


# Fit the model and validate with the validation set

history = model.fit_generator(train_generator, 
                              epochs=40,
                              validation_data = validation_generator,
                              callbacks=[learning_rate_scheduler], 
                              verbose = 1)

# save the trained model
model.save("skin_cancer.h5")

# Evaluate the model with the test examples

test_labels = test_generator.labels
test_probs = model.predict_generator(testing_generator) # the classification probabilities for each test instance
test_preds = test_probs.argmax(axis = -1)  # best test prediction according to the class with the highest probability

# Print the precision, recall, and F1 score for each class 

report = classification_report(test_labels,
                               test_best,
			       labels = np.arange(7))
print(report)			       
