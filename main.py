
import model
import model_softmax
import model_softmax_regularized
import lr
import h5py
import numpy as np
import pandas as pd

from dataLoader import generator
from Loss import KL_Distr

import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow import keras
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping
# from keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
import os

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#gpu = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpu[0], True)

all_cities = False
distributional = False
dry_run = False
distr_learning =True
urban = True
regularization = False

alpha = 0
prior_parameter = 0.5

batchSize=64
lrate = 0.0002
l2_var = 0.01

uncertain = False
entropy_quantile = 0

###################################################
'path to save models from check points:'
if all_cities:
    file0 = '/data/lcz42_votes/benchmark-on-So2SatLCZ42-dataset-a-simple-tour/results/all_cities/'
else:
    file0 = '/data/lcz42_votes/benchmark-on-So2SatLCZ42-dataset-a-simple-tour/results/'
file0 = 'C:/Users/kolle/PycharmProjects/Learning_Distributions_Sen2LCZ/results/'

'path to data, needs to be set accordingly'
if all_cities:
    train_file = '/data/lcz42_cities/train_data.h5'
    validation_file = '/data/lcz42_cities/validation_data.h5'
    path_data = "/data/lcz42_cities/"
#else:
#    train_file = '/data/lcz42_votes/data/train_data.h5'
#    validation_file = '/data/lcz42_votes/data/validation_data.h5'
#    path_data = "/data/lcz42_votes/data/"

train_file = 'D:/Data/LCZ_Votes/train_data.h5'
validation_file = 'D:/Data/LCZ_Votes/validation_data.h5'

if urban:
    mode = "urban"

train_data = h5py.File(train_file, 'r')
x_train = np.array(train_data.get("sen2"))
y_train = np.array(train_data.get("y"))

validation_data = h5py.File(validation_file, 'r')
x_val = np.array(validation_data.get("sen2"))
y_val = np.array(validation_data.get("y"))

if urban:
    indices_train = np.where(np.where(y_train == np.amax(y_train, 0))[1] + 1 < 11)[0]
    x_train = x_train[indices_train, :, :, :]
    indices_val = np.where(np.where(y_val == np.amax(y_val, 0))[1] + 1 < 11)[0]
    x_val = x_val[indices_val, :, :, :]

    if not distributional:
        y_train = y_train[indices_train]
        y_val = y_val[indices_val]

if entropy_quantile > 0 and urban:
    entropies = h5py.File(path_data + "entropies_train.h5", 'r')
    entropies_train = np.array(entropies.get("entropies_train"))
    entropies_train = entropies_train[indices_train]
    entropies_train[np.where(np.isnan(entropies_train))] = 0

    entropies = pd.DataFrame({"entropies": entropies_train,
                              "order": np.arange(len(y_train))})
    if not uncertain:
        entropies = entropies.sort_values(by=['entropies'])
    else:
        entropies = entropies.sort_values(by=['entropies'], ascending=False)
    ## Order training data accordingly
    idx = np.array(entropies["order"])
    ## Cut off at given quantile
    idx = idx[:np.floor(entropy_quantile * len(idx)).astype(int)]
    x_train = x_train[idx, :, :, :]
    y_train = y_train[idx]

if distributional:
    if alpha > 0:
        train_distributions = h5py.File('/data/lcz42_votes/data/train_label_distributions_data' + '_alpha_' + str(alpha) + '.h5', 'r')
        y_train = np.array(train_distributions['train_label_distributions'])
        val_distributions = h5py.File('/data/lcz42_votes/data/val_label_distributions_data' + '_alpha_' + str(alpha) + '.h5', 'r')
        y_val = np.array(val_distributions['val_label_distributions'])
    else:
        train_distributions = h5py.File('/data/lcz42_votes/data/train_label_distributions_data.h5', 'r')
        y_train = np.array(train_distributions['train_label_distributions'])
        val_distributions = h5py.File('/data/lcz42_votes/data/val_label_distributions_data.h5', 'r')
        y_val = np.array(val_distributions['val_label_distributions'])

if distr_learning:
    y_train = np.array(train_data['y_distributional_all'])
    y_val = np.array(validation_data['y_distributional_all'])
    if urban:
        y_train = y_train[indices_train]
        y_val = y_val[indices_val]

'number of all samples in training and validation sets'
trainNumber=y_train.shape[0]
validationNumber=y_val.shape[0]

lr_sched = lr.step_decay_schedule(initial_lr=lrate, decay_factor=0.5, step_size=5)

###################################################

if urban:
    model = model.sen2LCZ_drop(depth=17, dropRate=0.2, fusion=1, num_classes=10)
else:
    model = model.sen2LCZ_drop(depth=17, dropRate=0.2, fusion=1)

if distributional:
    model.compile(optimizer=Nadam(), loss='KLDivergence', metrics=['KLDivergence'])
else:
    model.compile(optimizer=Nadam(), loss='categorical_crossentropy', metrics=['accuracy'])

if dry_run:
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 100)
else:
    early_stopping = EarlyStopping(monitor='val_loss', patience=40)

if distr_learning:
    if regularization:
        model = model_softmax_regularized.sen2LCZ_drop(depth=17, dropRate=0.2, fusion=1, num_classes=10, l2_reg = l2_var)
    else:
        model = model_softmax.sen2LCZ_drop(depth=17, dropRate=0.2, fusion=1, num_classes=10)

    model.compile(optimizer=Nadam(), loss = KL_Distr, metrics=['accuracy'])

PATH = file0 + "Sen2LCZ_" + str(batchSize) + "_lr_" + str(lrate)
if urban:
    PATH = PATH + "_urban"

if distributional:
    if alpha > 0:
        PATH = PATH + "_d" + "alpha_" + str(alpha)
    else:
        PATH = PATH + "_d"

if distr_learning:
    PATH = PATH + "_dl"

if entropy_quantile > 0:
    if uncertain:
        PATH = PATH + "_most_uncertain_" + str(entropy_quantile)
    else:
        PATH = PATH + "_most_certain_" + str(entropy_quantile)

if dry_run:
    PATH = PATH + "_dry_run"

modelbest = PATH + "_weights_best.hdf5"

if distributional:
    checkpoint = ModelCheckpoint(modelbest, monitor='val_kullback_leibler_divergence', verbose=1, save_best_only=True,
                                 save_weights_only=True, mode='auto', save_freq='epoch')
else:
    checkpoint = ModelCheckpoint(modelbest, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=True, mode='auto', save_freq='epoch')

model.fit(generator(x_train, y_train, batchSize=batchSize, num=trainNumber, mode=mode),
                steps_per_epoch = trainNumber//batchSize,
                validation_data= generator(x_val, y_val, num=validationNumber, batchSize=batchSize, mode=mode),
                validation_steps = validationNumber//batchSize,
                epochs=100,
                max_queue_size=100,
                callbacks=[early_stopping, checkpoint, lr_sched])

