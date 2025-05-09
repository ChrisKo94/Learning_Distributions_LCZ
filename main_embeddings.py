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
from Loss import dirichlet_kl_divergence, mahala_dist_cov, mahala_dist_corr

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

## Set path ##
path = os.getcwd()
results_dir = Path(path, 'results')
results_dir.mkdir(parents=True, exist_ok=True)

np.random.seed(4242424242)

data_embedding = h5py.File('E:/Dateien/LCZ_Votes/embedding_data.h5', 'r')
patches = np.array(data_embedding.get("x"))
#labels = np.array(data_embedding.get("y")).astype(np.float32)

# import correlation matrix
corr_mat = pd.read_csv('E:/Dateien/LCZ_Votes/embeddings_correlation.csv')
corr_mat = np.array(corr_mat.drop(corr_mat.columns[0], axis=1))

#Train Val Test Split
shuffled_indices = np.random.permutation(patches.shape[0])
patches = patches[shuffled_indices,:,:,:]

# Temperature Scaling of exponentiated labels
#temperature = 3
#labels = np.exp(labels/temperature)

# Softmax Trafo
#labels = np.exp(labels) / np.sum(np.exp(labels), axis=1, keepdims=True)

train_patches, val_patches, test_patches = np.split(patches, [int(.6*len(patches)), int(.8*len(patches))])


trainNumber = train_patches.shape[0]
validationNumber = val_patches.shape[0]

########################################################################################################################
################################ Model Training ########################################################################
########################################################################################################################

def train_model(setting_dict: dict):
    # zeroes mean test, ones mean train

    seed = setting_dict["Seed"]

    if mode == "one-hot":
        model = model_with_softmax.sen2LCZ_drop(depth=17,
                                           dropRate=setting_dict["Data"]["dropout"],
                                           fusion=setting_dict["Data"]["fusion"],
                                           num_classes=setting_dict["Data"]["num_classes"])
        model.compile(optimizer=Nadam(),
                      loss=keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])
        labels = np.array(data_embedding.get("y_one_hot")).astype(np.float32)
    elif mode == "distributional":
        model = model_with_softmax.sen2LCZ_drop(depth=17,
                                                dropRate=setting_dict["Data"]["dropout"],
                                                fusion=setting_dict["Data"]["fusion"],
                                                num_classes=setting_dict["Data"]["num_classes"])
        model.compile(optimizer=Nadam(),
                      loss='KLDivergence',
                      metrics=['KLDivergence'])
        labels = np.array(data_embedding.get("y_distributional")).astype(np.float32)
    elif mode == 'sampled_one-hot':
        model = model_with_softmax.sen2LCZ_drop(depth=17,
                                                dropRate=setting_dict["Data"]["dropout"],
                                                fusion=setting_dict["Data"]["fusion"],
                                                num_classes=setting_dict["Data"]["num_classes"])
        model.compile(optimizer=Nadam(),
                      loss=keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])
        labels = np.array(data_embedding.get("y_distributional")).astype(np.float32)
    elif mode == "dirichlet":
        model = model_without_softmax.sen2LCZ_drop(depth=17,
                                                dropRate=setting_dict["Data"]["dropout"],
                                                fusion=setting_dict["Data"]["fusion"],
                                                num_classes=setting_dict["Data"]["num_classes"])
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
                                                   num_classes=setting_dict["Data"]["num_classes"])
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
                                                   num_classes=setting_dict["Data"]["num_classes"])
        model.compile(optimizer=Nadam(),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[tf.keras.losses.MeanSquaredError()])
        labels = np.array(data_embedding.get("y")).astype(np.float32)
    elif mode == "Mahala_embedding":
        model = model_without_softmax.sen2LCZ_drop(depth=17,
                                                   dropRate=setting_dict["Data"]["dropout"],
                                                   fusion=setting_dict["Data"]["fusion"],
                                                   num_classes=setting_dict["Data"]["num_classes"])
        model.compile(optimizer=Nadam(),
                      loss=mahala_dist_corr,
                      metrics=[mahala_dist_corr])
        labels = np.array(data_embedding.get("y")).astype(np.float32)
    print("Model compiled")

    labels = labels[shuffled_indices, :]
    train_labels, val_labels, test_labels = np.split(labels, [int(.6 * len(labels)), int(.8 * len(labels))])

    batchSize = setting_dict["Data"]["train_batch_size"]
    lrate = setting_dict["Optimization"]["lr"]

    lr_sched = lr.step_decay_schedule(initial_lr=lrate,
                                      decay_factor=0.5,
                                      step_size=5)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=setting_dict["Optimization"]["patience"])
    ckpt_file = Path(path, "results", f"Sen2LCZ_bs_{batchSize}_lr_{lrate}_seed_{seed}_weights_best_{mode}_random_split_30_cv_5.hdf5")

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
        model.fit(generator_sampled(train_patches,
                                    train_labels,
                                    batchSize=batchSize,
                                    num=trainNumber),
                  steps_per_epoch=trainNumber // batchSize,
                  validation_data=generator_sampled(val_patches,
                                                    val_labels,
                                                    num=validationNumber,
                                                    batchSize=batchSize),
                  validation_steps=validationNumber // batchSize,
                  epochs=setting_dict["Trainer"]["max_epochs"],
                  max_queue_size=100,
                  callbacks=[early_stopping, checkpoint, lr_sched])
    else:
        model.fit(generator(train_patches,
                            train_labels,
                            batchSize=batchSize,
                            num=trainNumber),
                  steps_per_epoch=trainNumber // batchSize,
                  validation_data=generator(val_patches,
                                            val_labels,
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
    for mode in ["one-hot", "distributional","sampled_one-hot", "dirichlet", "Dirichlet_embedding","MSE_embedding", "Mahala_embedding"]: #
        for seed in range(1):
            setting_dict["Seed"] = seed
            setting_dict["Data"]["mode"] = mode
            train_model(setting_dict)