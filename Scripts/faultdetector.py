#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb 14 17:05:43 2019

@author: cros_xa
"""

""" -*- LIBRAIRIES -*- """

import math
import shutil
import os, os.path
import matplotlib.pyplot as plt
import numpy             as np

np.random.seed()

from keras.models                      import Sequential
from keras.layers                      import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image         import ImageDataGenerator

""" -*- DIRECTORIES -*- """

dir_main    = "/home/cros_x/Documents/FautDetector"
dir_data    = dir_main + "/Data"
dir_plots   = dir_main + "/Plots"

os.chdir(dir_main)

""" -*- SPLIT DATA -*- """

if os.path.isdir(dir_data  + "/train"):
    shutil.rmtree(dir_data + "/train")
    print("previous training set removed")
if os.path.isdir(dir_data  + "/test"):
    shutil.rmtree(dir_data + "/test")
    print("previous testing set removed")

os.mkdir(dir_data + "/train")
os.mkdir(dir_data + "/test")
os.mkdir(dir_data + "/train/defect")
os.mkdir(dir_data + "/train/good")
os.mkdir(dir_data + "/test/defect")
os.mkdir(dir_data + "/test/good")
print("folder tree completed")

defect = sorted(os.listdir(dir_data + "/defect"))
defect = np.asarray(defect)

_ = np.repeat([True, False], [math.floor(len(defect)*.8), len(defect)-math.floor(len(defect)*.8)])

defect_train = defect[ _] ; defect_train = defect_train.tolist()
defect_test  = defect[~_] ; defect_test  = defect_test.tolist()
    
if len(defect_test) + len(defect_train) != len(defect):
    print("error len defect test/train samples ...")
    
good = sorted(os.listdir(dir_data + "/good"))
good = np.asarray(good)

_ = np.repeat([True, False], [math.floor(len(good)*.8), len(good)-math.floor(len(good)*.8)])

good_train = good[ _] ; good_train = good_train.tolist()
good_test  = good[~_] ; good_test  = good_test.tolist()
    
if len(good_test) + len(good_train) != len(good):
    print("error len good test/train samples ...")
    
print("copying images to train/test folders ...")

for file_name in defect_train:
    file_path = os.path.join(dir_data + "/defect", file_name)
    if os.path.isfile(file_path):
        shutil.copy(file_path, dir_data + "/train/defect")

for file_name in defect_test:
    file_path = os.path.join(dir_data + "/defect", file_name)
    if os.path.isfile(file_path):
        shutil.copy(file_path, dir_data + "/test/defect")

for file_name in good_train:
    file_path = os.path.join(dir_data + "/good", file_name)
    if os.path.isfile(file_path):
        shutil.copy(file_path, dir_data + "/train/good")

for file_name in good_test:
    file_path = os.path.join(dir_data + "/good", file_name)
    if os.path.isfile(file_path):
        shutil.copy(file_path, dir_data + "/test/good")

print("train/test set splitted")

""" -*- CNN STRUCTURE -*- """

# Init
fault_classifier = Sequential()

# Conv layers
n_conv = 3 ; n_kernel = 32
for _ in range(n_conv):
    fault_classifier.add(
        Convolution2D(
            n_kernel, 3, 3, 
            input_shape = (150, 150, 3), 
            activation  = 'relu'
        )
    )
    fault_classifier.add(
        MaxPooling2D(
            pool_size = (2, 2)
        )
    )
    n_kernel *=2

del n_conv, n_kernel

# Flatten
fault_classifier.add(
    Flatten()
)

# Dense layers
fault_classifier.add(
    Dense(
        output_dim = 128, 
        activation = 'relu'
    )
)
fault_classifier.add(
    Dense(
        output_dim = 1, 
        activation = 'sigmoid'
    )
)

# Compile
fault_classifier.compile(
    optimizer = 'adam', 
    loss      = 'binary_crossentropy', 
    metrics   = ['accuracy']
)

""" -*- CNN-FIT -*- """

# Data augmentation (prevents overfitrting)
train_datagenerator = ImageDataGenerator(
    rescale         = 1./255,
    shear_range     = 0.2,
    zoom_range      = 0.2,
    horizontal_flip = True
)

# randon modification on images
test_datagenerator = ImageDataGenerator(
    rescale = 1./255
)

# data
train_generator = train_datagenerator.flow_from_directory(
    dir_data + "/train",
    target_size = (150, 150),
    batch_size  = 32,
    class_mode  = 'binary'
)
test_generator = test_datagenerator.flow_from_directory(
    dir_data + "/test",
    target_size = (150, 150),
    batch_size  = 32,
    class_mode  = 'binary'
)

print(fault_classifier.summary())

# Fit
fault_classifier.fit_generator(
    train_generator,
    samples_per_epoch = 2000,
    nb_epoch          = 25,
    validation_data   = test_generator,
    validation_steps  = 800
)

""" -*- TRAIN HISTORY VISUALIZATION -*- """

model = fault_classifier.fit_generator(
    train_generator,
    samples_per_epoch = 2000,
    nb_epoch          = 25,
    validation_data   = test_generator,
    validation_steps  = 800
)

# training & validation accuracy values
plt.plot(model.history['acc'])
plt.plot(model.history['val_acc'])
plt.ylim(0, 1.1)
plt.title('Accuracy')
plt.xlabel('Epoch') ; plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc = 'upper left')
plt.show()
plt.savefig(dir_plots + '/CNNmodel_acc.png')

# training & validation loss values
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('Loss')
plt.xlabel('Epoch') ; plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc = 'upper left')
plt.show()
plt.savefig(dir_plots + '/CNNmodel_loss.png')

""" -*- END -*- """
