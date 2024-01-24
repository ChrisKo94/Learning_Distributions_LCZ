import model_softmax
#import model
import lr
import numpy as np
import pandas as pd
import h5py

from dataLoader import generator
from Loss import dirichlet_kl_divergence

from utils import *

import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow import keras
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping
# from keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
import os

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
gpu = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpu[0], True)

np.random.seed(42)

#Todo: Automatic Data read-in from h5 file

data_embedding = h5py.File('D:/Data/LCZ_Votes/embedding_data.h5', 'r')
patches = np.array(data_embedding.get("x"))
labels = np.array(data_embedding.get("y"))

#Train Val Test Split
shuffled_indices = np.random.permutation(patches.shape[0])
patches = patches[shuffled_indices,:,:,:]
labels = labels[shuffled_indices,:]

# Temperature Scaling of exponentiated labels
temperature = 1
labels = np.exp(labels/temperature)

# Softmax Trafo
#labels = np.exp(labels) / np.sum(np.exp(labels), axis=1, keepdims=True)

train_patches, val_patches, test_patches = np.split(patches, [int(.6*len(patches)), int(.8*len(patches))])
train_labels, val_labels, test_labels = np.split(labels, [int(.6*len(labels)), int(.8*len(labels))])

########################################################################################################################
################################ Model Training ########################################################################
########################################################################################################################

# Todo: def train_model based on setting_dict
# Todo: training loop for multiple seeds or configs

batchSize=64
lrate = 0.0002

file0 = 'C:/Users/kolle/PycharmProjects/Learning_Distributions_LCZ/results/embeddings/'

model = model_softmax.sen2LCZ_drop(depth=17, dropRate=0.2, fusion=1, num_classes=16)

#model.compile(optimizer=Nadam(), loss='KLDivergence', metrics=['KLDivergence'])
model.compile(optimizer=Nadam(),
              loss=dirichlet_kl_divergence, #Todo: Check following error: NotImplementedError: Cannot convert a symbolic Tensor (dirichlet_kl_divergence/truediv:0) to a numpy array. This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported
              #loss=tf.keras.losses.MeanSquaredError(),
              #metrics=[keras.metrics.mean_squared_error,
              #         keras.metrics.mean_absolute_error]
                       )

lr_sched = lr.step_decay_schedule(initial_lr=lrate, decay_factor=0.5, step_size=5)
early_stopping = EarlyStopping(monitor='val_loss', patience=40)

PATH = file0 + "Sen2LCZ_" + str(batchSize) + "_lr_" + str(lrate)
modelbest = PATH + "_weights_best.hdf5"

#checkpoint = ModelCheckpoint(modelbest, monitor='val_kullback_leibler_divergence', verbose=1, save_best_only=True,
#                             save_weights_only=True, mode='auto', save_freq='epoch')
checkpoint = ModelCheckpoint(modelbest, monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=True, mode='auto', save_freq='epoch')
'''
# Try with subset

#cities_patches = cities_patches[:1000,:,:,:]
#z_all_images = z_all_images[:1000,:]

# Try with original labels

train_file = 'E:/Dateien/LCZ_Votes/train_data.h5'
train_data = h5py.File(train_file, 'r')
x_train = np.array(train_data.get("sen2"))
y_train = np.array(train_data.get("y"))
y_train = y_train[:1000,:]
z_all_images = y_train
#cities_patches = x_train[:1000,:,:,:]

validation_file = 'E:/Dateien/LCZ_Votes/validation_data.h5'
validation_data = h5py.File(validation_file, 'r')
x_val = np.array(validation_data.get("sen2"))
y_val = np.array(validation_data.get("y"))
'''
trainNumber=train_patches.shape[0]
validationNumber=val_patches.shape[0]

model.fit(generator(train_patches, train_labels, batchSize=batchSize, num=trainNumber),
                steps_per_epoch = trainNumber//batchSize,
                validation_data= generator(val_patches, val_labels, num=validationNumber, batchSize=batchSize),
                validation_steps = validationNumber//batchSize,
                epochs=100,
                max_queue_size=100,
                callbacks=[lr_sched])
                #callbacks=[early_stopping, checkpoint, lr_sched])

# Todo: Save model weights
# Todo: Calibration analysis (here or separate script)