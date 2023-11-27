import model_softmax
import model
import lr
import numpy as np
import pandas as pd
import h5py

from dataLoader import generator
from Loss import KL_Distr

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
tf.config.experimental.set_memory_growth(gpu[0], True)


########################################################################################################################
################################ Data Generation (matching with Katharina's code) ######################################
########################################################################################################################

city_list = ['berlin', 'cologne', 'london', 'madrid', 'milan', 'munich', 'paris', 'rome', 'zurich']

# Define labels
labels = np.arange(1,18)

cities_frames = [process_city(city) for city in city_list]
cities_votes_named = pd.concat(cities_frames)

cities_votes = pd.DataFrame(concatenate_cities(city_list).astype(int))

# to one hot

cities_one_hot = pd.DataFrame(to_one_hot(cities_votes, labels))
cities_one_hot_named = cities_one_hot.copy()
cities_one_hot_named['City'] = np.array(cities_votes_named['City'])

# Extract patches

cities_patches = concatenate_cities_patches(city_list)
indeces_out = np.array(cities_one_hot[cities_one_hot[6] != 0].index)
cities_patches = np.delete(cities_patches, indeces_out, 0)
# delete instances with vote for class 7

cities_one_hot_16 = cities_one_hot.drop(cities_one_hot[cities_one_hot[6] != 0].index)
cities_one_hot_16 = cities_one_hot_16.drop(columns=[6])

katharina_embeddings = pd.read_csv("E:/Downloads/Archiv/df_z_full_all_img.csv")

# 2. unique patterns with frequency for model ----------------------

vote_patterns = np.unique(cities_one_hot_16, axis=0, return_inverse=True, return_counts=True)

# 3. import csv of embeddings from R -------------------------------

z_hat = pd.read_csv('E:/Downloads/Archiv/z_full.csv', ) #Todo: Change path to project dir

z_hat = np.array(z_hat.drop(z_hat.columns[0], axis=1))

mapping = vote_patterns[1]
z_all_images = []
for m in mapping:
    z_all_images.append(z_hat[m])

z_all_images = np.array(z_all_images)

########################################################################################################################
################################ Model Training ########################################################################
########################################################################################################################

batchSize=64
lrate = 0.0002

file0 = 'C:/Users/kolle/PycharmProjects/Learning_Distributions_LCZ/results/embeddings/'

model = model.sen2LCZ_drop(depth=17, dropRate=0.2, fusion=1, num_classes=16)

model.compile(optimizer=Nadam(), loss='KLDivergence', metrics=['KLDivergence'])

lr_sched = lr.step_decay_schedule(initial_lr=lrate, decay_factor=0.5, step_size=5)
early_stopping = EarlyStopping(monitor='val_loss', patience=40)

PATH = file0 + "Sen2LCZ_" + str(batchSize) + "_lr_" + str(lrate)
modelbest = PATH + "_weights_best.hdf5"

checkpoint = ModelCheckpoint(modelbest, monitor='val_kullback_leibler_divergence', verbose=1, save_best_only=True,
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
trainNumber=cities_patches.shape[0]
#validationNumber=y_val.shape[0]

model.fit(generator(cities_patches, z_all_images, batchSize=batchSize, num=trainNumber),
                steps_per_epoch = trainNumber//batchSize,
                #validation_data= generator(x_val, y_val, num=validationNumber, batchSize=batchSize),
                #validation_steps = validationNumber//batchSize,
                epochs=100,
                max_queue_size=100,
                callbacks=[lr_sched])
                #callbacks=[early_stopping, checkpoint, lr_sched])