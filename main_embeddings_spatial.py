import model_softmax
#import model
import lr
import numpy as np
import pandas as pd
import h5py

from dataLoader import generator
from Loss import dirichlet_kl_divergence, mahala_dist

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

city_list = ['berlin', 'cologne', 'london', 'madrid', 'milan', 'munich', 'paris', 'rome', 'zurich']

for city in city_list:
    name_tmp = "E:/Dateien/LCZ_Votes/" + city + "_geo.csv"
    city_tmp=pd.read_csv(name_tmp, )
    city_tmp = np.array(city_tmp['Lon'])
    percentile = np.percentile(city_tmp, 80)
    # mark as train (=1) if below percentile and as test (=0) if above
    city_tmp = np.where(city_tmp < percentile, 1, 0)
    if city == city_list[0]:
        cities_geo = city_tmp
    else:
        cities_geo = np.hstack((cities_geo, city_tmp))

    cities_geo = np.array(cities_geo)

'''
# Delete instances with votes for class 7

# Define labels
labels = np.arange(1,18)

cities_frames = [process_city(city) for city in city_list]
cities_votes_named = pd.concat(cities_frames)

cities_votes = pd.DataFrame(concatenate_cities(city_list).astype(int))

# to one hot

cities_one_hot = pd.DataFrame(to_one_hot(cities_votes, labels))

indeces_out = np.array(cities_one_hot[cities_one_hot[6] != 0].index)
'''

ind_out = np.array([ 16455,  16456,  16457,  16495,  16496,  16497,  16498,  16499,
        16500,  16537,  16538,  16539,  16540,  16541,  16542,  16543,
        16577,  16578,  16579,  16580,  16581,  16582,  16583,  16604,
        16605,  16606,  16607,  16608,  16609,  16629,  16630,  16631,
        16632,  16633,  16634,  16657,  16658,  16659,  16660,  16661,
        16662,  16688,  16689,  16690,  16691,  16692,  16732,  16733,
        16734,  16735,  16777,  16778,  16779,  16780,  16829,  16830,
        16831,  16832,  16883,  16884,  16885,  16935, 105867, 105868,
       105869, 105905, 105906, 105907, 106063, 106064, 106065, 106066,
       106101, 106102, 106103, 106126, 106127, 106128, 106129, 106130,
       106131, 106159, 106160, 106161, 106162, 106539, 106546, 106848,
       106865, 106866, 156806, 156820, 156821, 156822, 156839, 156840,
       156841, 156842, 156856, 156857, 156913, 156914, 156915, 156931,
       156932, 156933, 156934, 156935, 156936, 156944, 156945, 156946,
       156947, 156948, 156949, 156950, 156951, 156965, 156966, 156967,
       156968, 156969, 156970, 156985, 156986, 158173, 158210, 158249,
       158250, 158286, 158287, 158320, 158321, 158350, 158351])

cities_geo = np.delete(cities_geo, ind_out, 0)

# zeroes mean test, ones mean train
data_embedding = h5py.File('E:/Dateien/LCZ_Votes/embedding_data.h5', 'r')
patches = np.array(data_embedding.get("x"))
labels = np.array(data_embedding.get("y")).astype(np.float32)

# import correlation matrix
corr_mat = pd.read_csv('E:/Dateien/LCZ_Votes/embeddings_correlation.csv')
corr_mat = np.array(corr_mat.drop(corr_mat.columns[0], axis=1))

# Temperature Scaling of exponentiated labels
#temperature = 3
#labels = np.exp(labels/temperature)

patches_train = patches[cities_geo == 1]
labels_train = labels[cities_geo == 1]

patches_test = patches[cities_geo == 0]
labels_test = labels[cities_geo == 0]

#Train Val Test Split
shuffled_indices_train = np.random.permutation(patches_train.shape[0])
patches_train = patches_train[shuffled_indices_train,:,:,:]
labels_train = labels_train[shuffled_indices_train,:]

shuffled_indices_test = np.random.permutation(patches_test.shape[0])
patches_test = patches_test[shuffled_indices_test,:,:,:]
labels_test = labels_test[shuffled_indices_test,:]

# split train into train and val
train_patches, val_patches = np.split(patches_train, [int(.75*len(patches_train))])
train_labels, val_labels = np.split(labels_train, [int(.75*len(labels_train))])

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
              #loss=dirichlet_kl_divergence, metrics=[dirichlet_kl_divergence]
              loss= mahala_dist, metrics=[mahala_dist]
              #loss=tf.keras.losses.MeanSquaredError(),
              #loss=tf.keras.losses.CategoricalCrossentropy(),
              #loss=tf.keras.losses.KLDivergence(),
              #metrics=[keras.metrics.mean_squared_error,
              #         keras.metrics.mean_absolute_error]
                       )

lr_sched = lr.step_decay_schedule(initial_lr=lrate, decay_factor=0.5, step_size=5)
early_stopping = EarlyStopping(monitor='val_loss', patience=50)

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
                epochs=20,
                max_queue_size=100,
                #callbacks=[lr_sched])
                callbacks=[early_stopping, checkpoint, lr_sched])

test_preds = model.predict(patches_test)
test_preds_labels = np.argmax(test_preds, axis=1)
test_labels_one_hot = np.argmax(labels_test, axis=1)
test_acc = np.mean(test_preds_labels == test_labels_one_hot)

# Validation accuracy
val_preds = model.predict(val_patches)
val_preds_labels = np.argmax(val_preds, axis=1)
val_labels_one_hot = np.argmax(val_labels, axis=1)
val_acc = np.mean(val_preds_labels == val_labels_one_hot)

# Test ECE
test_preds_softmax = np.exp(test_preds) / np.sum(np.exp(test_preds), axis=1, keepdims=True)
test_ece = ECE(test_preds_softmax, test_labels_one_hot, 10)

# Validation ECE
val_preds_softmax = np.exp(val_preds) / np.sum(np.exp(val_preds), axis=1, keepdims=True)
val_ece = ECE(val_preds_softmax, val_labels_one_hot, 10)

# Test ECE (with trafo)
test_preds_softmax = np.exp(test_preds*3) / np.sum(np.exp(test_preds*3), axis=1, keepdims=True)
test_ece = ECE(test_preds_softmax, test_labels_one_hot, 10)

# Validation ECE (with trafo)
val_preds_softmax = np.exp(val_preds*3) / np.sum(np.exp(val_preds*3), axis=1, keepdims=True)
val_ece = ECE(val_preds_softmax, val_labels_one_hot, 10)
