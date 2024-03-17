import model_softmax
#import model
import lr
import numpy as np
import pandas as pd
import h5py

from dataLoader import generator
from Loss import dirichlet_kl_divergence
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from utils import *

import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow import keras
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping
# from keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
import os

import matplotlib.pyplot as plt

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
gpu = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpu[0], True)

np.random.seed(42)

data_embedding = h5py.File('E:/Dateien/LCZ_Votes/embedding_data.h5', 'r')
patches = np.array(data_embedding.get("x"))
labels = np.array(data_embedding.get("y"))
one_hot_labels = np.array(data_embedding.get("y_one_hot"))

# ID and OOD Split
# ID: 0-9, OOD: 10-16
id_indices = np.where(one_hot_labels[:,0:9] == 1)[0]
ood_indices = np.where(one_hot_labels[:,9:16] == 1)[0]


#Train Val Test Split ID
patches_id = patches[id_indices,:,:,:]
labels_id = labels[id_indices,:9]

shuffled_indices_id = np.random.permutation(patches_id.shape[0])
patches_id = patches_id[shuffled_indices_id,:,:,:]
labels_id = labels_id[shuffled_indices_id,:]

#Train Val Test Split OOD
patches_ood = patches[ood_indices,:,:,:]
labels_ood = labels[ood_indices,9:16]

shuffled_indices_ood = np.random.permutation(patches_ood.shape[0])
patches_ood = patches_ood[shuffled_indices_ood,:,:,:]
labels_ood = labels_ood[shuffled_indices_ood,:]

# Temperature Scaling of exponentiated labels
temperature = 3
labels_id = np.exp(labels_id/temperature)
labels_ood = np.exp(labels_ood/temperature)

# Softmax Trafo
#labels = np.exp(labels) / np.sum(np.exp(labels), axis=1, keepdims=True)

train_patches_id, val_patches_id, test_patches_id = np.split(patches_id, [int(.6*len(patches_id)), int(.8*len(patches_id))])
train_patches_ood, val_patches_ood, test_patches_ood = np.split(patches_ood, [int(.6*len(patches_ood)), int(.8*len(patches_ood))])

train_labels_id, val_labels_id, test_labels_id = np.split(labels_id, [int(.6*len(labels_id)), int(.8*len(labels_id))])
train_labels_ood, val_labels_ood, test_labels_ood = np.split(labels_ood, [int(.6*len(labels_ood)), int(.8*len(labels_ood))])

batchSize=64
lrate = 0.0002

file0 = 'C:/Users/kolle/PycharmProjects/Learning_Distributions_LCZ/results/embeddings/'

model = model_softmax.sen2LCZ_drop(depth=17, dropRate=0.2, fusion=1, num_classes=9)

#model.compile(optimizer=Nadam(), loss='KLDivergence', metrics=['KLDivergence'])
model.compile(optimizer=Nadam(),
              loss=dirichlet_kl_divergence, metrics=[dirichlet_kl_divergence]
              #loss=tf.keras.losses.MeanSquaredError(),
              #loss=tf.keras.losses.CategoricalCrossentropy(),
              #loss=tf.keras.losses.KLDivergence(),
              #metrics=[keras.metrics.mean_squared_error,
              #         keras.metrics.mean_absolute_error]
                       )

lr_sched = lr.step_decay_schedule(initial_lr=lrate, decay_factor=0.5, step_size=5)
early_stopping = EarlyStopping(monitor='val_loss', patience=50)

PATH = file0 + "Sen2LCZ_" + str(batchSize) + "_lr_" + str(lrate)
modelbest = PATH + "_weights_best_urban_id_train.hdf5"

#checkpoint = ModelCheckpoint(modelbest, monitor='val_kullback_leibler_divergence', verbose=1, save_best_only=True,
#                             save_weights_only=True, mode='auto', save_freq='epoch')
checkpoint = ModelCheckpoint(modelbest, monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=True, mode='auto', save_freq='epoch')

trainNumber=train_patches_id.shape[0]
validationNumber=val_patches_id.shape[0]

model.fit(generator(train_patches_id, train_labels_id, batchSize=batchSize, num=trainNumber),
                steps_per_epoch = trainNumber//batchSize,
                validation_data= generator(val_patches_id, val_labels_id, num=validationNumber, batchSize=batchSize),
                validation_steps = validationNumber//batchSize,
                epochs=30,
                max_queue_size=100,
                #callbacks=[lr_sched])
                callbacks=[early_stopping, checkpoint, lr_sched])

id_test_preds = model.predict(test_patches_id)
ood_test_preds = model.predict(test_patches_ood)
id_test_preds = id_test_preds[:10000,:]
ood_test_preds = ood_test_preds[:10000,:]

id_params = np.exp(id_test_preds)
id_params_scaled = np.exp(id_test_preds*temperature)

ood_params = np.exp(ood_test_preds)
ood_params_scaled = np.exp(ood_test_preds*temperature)

id_variances = dirichlet_variance(np.exp(id_test_preds))


ood_variances = dirichlet_variance(np.exp(ood_test_preds))

# boxplot of id variances

plt.boxplot([id_variances[:,0], id_variances[:,1], id_variances[:,2], id_variances[:,3], id_variances[:,4],
                id_variances[:,5], id_variances[:,6], id_variances[:,7], id_variances[:,8]])
plt.show()

# boxplot of ood variances

plt.boxplot([ood_variances[:,0], ood_variances[:,1], ood_variances[:,2], ood_variances[:,3], ood_variances[:,4],
                ood_variances[:,5], ood_variances[:,6], ood_variances[:,7], ood_variances[:,8]])
plt.show()