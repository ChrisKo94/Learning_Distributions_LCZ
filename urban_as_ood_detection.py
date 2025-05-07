import os
import argparse
import yaml
import random as rn
from numpy import random
from pathlib import Path
import model_without_softmax
import model_with_softmax
#import model
import lr
import numpy as np
import pandas as pd
import h5py
import gc

from dataLoader import generator, generator_sampled
from Loss import dirichlet_kl_divergence, mahala_dist_corr_veg
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import scipy
import scipy.stats
import scipy.special

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

## Set path ##
path = os.getcwd()
results_dir = Path(path, 'results')
results_dir.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
# Todo: Change data dir
data_embedding = h5py.File('E:/Dateien/LCZ_Votes/embedding_data.h5', 'r')
patches = np.array(data_embedding.get("x"))

labels_for_separation = np.array(data_embedding.get("y_distributional"))

# ID and OOD Split
# ID: 0-9, OOD: 10-16
#id_indices = np.where(one_hot_labels[:,0:9] == 1)[0]
#ood_indices = np.where(one_hot_labels[:,9:16] == 1)[0]

id_indices=np.where(np.argmax(labels_for_separation,1)>8)[0]
ood_indices=np.where(np.argmax(labels_for_separation, 1) < 9)[0]
#Train Val Test Split ID
patches_id = patches[id_indices,:,:,:]

np.random.seed(42)
shuffled_indices_id = np.random.permutation(patches_id.shape[0])
patches_id = patches_id[shuffled_indices_id,:,:,:]

#Train Val Test Split OOD
patches_ood = patches[ood_indices,:,:,:]

np.random.seed(42)
shuffled_indices_ood = np.random.permutation(patches_ood.shape[0])
patches_ood = patches_ood[shuffled_indices_ood,:,:,:]

train_patches_id, val_patches_id, test_patches_id = np.split(patches_id, [int(.6*len(patches_id)), int(.8*len(patches_id))])
train_patches_ood, val_patches_ood, test_patches_ood = np.split(patches_ood, [int(.6*len(patches_ood)), int(.8*len(patches_ood))])

trainNumber=train_patches_id.shape[0]
validationNumber=val_patches_id.shape[0]

def train_model(setting_dict: dict):
    # zeroes mean test, ones mean train

    seed = setting_dict["Seed"]

    if mode == "one-hot":
        model = model_with_softmax.sen2LCZ_drop(depth=17,
                                           dropRate=setting_dict["Data"]["dropout"],
                                           fusion=setting_dict["Data"]["fusion"],
                                           num_classes=7)
        model.compile(optimizer=Nadam(),
                      loss=keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])
        labels = np.array(data_embedding.get("y_one_hot")).astype(np.float32)
    elif mode == "distributional":
        model = model_with_softmax.sen2LCZ_drop(depth=17,
                                                dropRate=setting_dict["Data"]["dropout"],
                                                fusion=setting_dict["Data"]["fusion"],
                                                num_classes=7)
        model.compile(optimizer=Nadam(),
                      loss='KLDivergence',
                      metrics=['KLDivergence'])
        labels = np.array(data_embedding.get("y_distributional")).astype(np.float32)
    elif mode == 'sampled_one-hot':
        model = model_with_softmax.sen2LCZ_drop(depth=17,
                                                dropRate=setting_dict["Data"]["dropout"],
                                                fusion=setting_dict["Data"]["fusion"],
                                                num_classes=7)
        model.compile(optimizer=Nadam(),
                      loss=keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])
        labels = np.array(data_embedding.get("y_distributional")).astype(np.float32)
    elif mode == "dirichlet":
        model = model_without_softmax.sen2LCZ_drop(depth=17,
                                                dropRate=setting_dict["Data"]["dropout"],
                                                fusion=setting_dict["Data"]["fusion"],
                                                num_classes=7)
        model.compile(optimizer=Nadam(),
                      loss=dirichlet_kl_divergence,
                      metrics=[dirichlet_kl_divergence])
        labels = np.array(data_embedding.get("y_distributional")).astype(np.float32)
        # 11 vote counts -> * 11, c=1 -> + 1
        labels = labels*11 + 1
        # Temperature Scaling of exponentiated labels w/ temperature = 3
        labels = np.exp(labels / 3)
    elif mode == "Dirichlet_embedding":
        model = model_without_softmax.sen2LCZ_drop(depth=17,
                                                   dropRate=setting_dict["Data"]["dropout"],
                                                   fusion=setting_dict["Data"]["fusion"],
                                                   num_classes=7)
        model.compile(optimizer=Nadam(),
                      loss=dirichlet_kl_divergence,
                      metrics=[dirichlet_kl_divergence])
        labels = np.array(data_embedding.get("y")).astype(np.float32)
        # Temperature Scaling of exponentiated labels w/ temperature = 3
        labels = np.exp(labels/3)

    elif mode == "MSE_embedding":
        model = model_without_softmax.sen2LCZ_drop(depth=17,
                                                   dropRate=setting_dict["Data"]["dropout"],
                                                   fusion=setting_dict["Data"]["fusion"],
                                                   num_classes=7)
        model.compile(optimizer=Nadam(),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[tf.keras.losses.MeanSquaredError()])
        labels = np.array(data_embedding.get("y")).astype(np.float32)
    elif mode == "Mahala_embedding":
        model = model_without_softmax.sen2LCZ_drop(depth=17,
                                                   dropRate=setting_dict["Data"]["dropout"],
                                                   fusion=setting_dict["Data"]["fusion"],
                                                   num_classes=7)
        model.compile(optimizer=Nadam(),
                      loss=mahala_dist_corr_veg,
                      metrics=[mahala_dist_corr_veg])
        labels = np.array(data_embedding.get("y")).astype(np.float32)
    print("Model compiled")

    labels_id = labels[id_indices, 9:16]
    # Make sure that rowsums are 1
    labels_id = labels_id / np.sum(labels_id, axis=1)[:, np.newaxis]
    labels_id = labels_id[shuffled_indices_id, :]
    labels_ood = labels[ood_indices, :9]
    labels_ood = labels_ood[shuffled_indices_ood, :]

    train_labels_id, val_labels_id, test_labels_id = np.split(labels_id,
                                                              [int(.6 * len(labels_id)), int(.8 * len(labels_id))])
    train_labels_ood, val_labels_ood, test_labels_ood = np.split(labels_ood,
                                                                 [int(.6 * len(labels_ood)), int(.8 * len(labels_ood))])

    batchSize = setting_dict["Data"]["train_batch_size"]
    lrate = setting_dict["Optimization"]["lr"]

    lr_sched = lr.step_decay_schedule(initial_lr=lrate,
                                      decay_factor=0.5,
                                      step_size=5)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=setting_dict["Optimization"]["patience"])
    ckpt_file = Path(path, "results", f"Sen2LCZ_bs_{batchSize}_lr_{lrate}_seed_{seed}_weights_best_{mode}_veg_as_id_short.hdf5")

    checkpoint = ModelCheckpoint(
        ckpt_file,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        save_freq='epoch')

    print("Callbacks and checkpoint initialized")

    ## Reproducibility
    random.seed(seed)
    rn.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = '0'

    if mode == "sampled_one-hot":
        model.fit(generator_sampled(train_patches_id,
                                    train_labels_id,
                                    batchSize=batchSize,
                                    num=trainNumber),
                  steps_per_epoch=trainNumber // batchSize,
                  validation_data=generator_sampled(val_patches_id,
                                                    val_labels_id,
                                                    num=validationNumber,
                                                    batchSize=batchSize),
                  validation_steps=validationNumber // batchSize,
                  epochs=setting_dict["Trainer"]["max_epochs"],
                  max_queue_size=100,
                  callbacks=[early_stopping, checkpoint, lr_sched])
    else:
        model.fit(generator(train_patches_id,
                            train_labels_id,
                            batchSize=batchSize,
                            num=trainNumber),
                  steps_per_epoch=trainNumber // batchSize,
                  validation_data=generator(val_patches_id,
                                            val_labels_id,
                                            num=validationNumber,
                                            batchSize=batchSize),
                  validation_steps=validationNumber // batchSize,
                  epochs=setting_dict["Trainer"]["max_epochs"],
                  max_queue_size=100,
                  callbacks=[early_stopping, checkpoint, lr_sched])

    gc.collect()

## Load settings dictionary ##

with open("configs/model_settings.yaml", 'r') as fp:
    setting_dict = yaml.load(fp, Loader=yaml.FullLoader)

parser = argparse.ArgumentParser()
parser.add_argument('--single_run', type=bool, required=False, default = False)
args = parser.parse_args()

## Train models ##

if __name__ == "__main__":
    for mode in ["one-hot", "distributional"]: # , "sampled_one-hot", "dirichlet", "Dirichlet_embedding", "MSE_embedding","Mahala_embedding"
        for seed in range(1,5):
            setting_dict["Seed"] = seed
            setting_dict["Data"]["mode"] = mode
            train_model(setting_dict)

'''
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


# Sampling pi's

sampled_pis_id = np.array([])

for i in range(len(id_params_scaled)):
    sampled_pis_id = np.append(sampled_pis_id, scipy.stats.entropy(np.random.dirichlet(id_params_scaled[i])))


sampled_pis_ood = np.array([])

for i in range(len(ood_params_scaled)):
    sampled_pis_ood = np.append(sampled_pis_ood, scipy.stats.entropy(np.random.dirichlet(ood_params_scaled[i])))

# Plot histogram of sampled pi's id and ood

plt.hist(sampled_pis_id, bins=100, alpha=0.5, label='ID')
plt.hist(sampled_pis_ood, bins=100, alpha=0.5, label='OOD')
plt.legend(loc='upper right')
plt.show()
'''