# @Date:   2020-02-04T16:58:21+01:00
# @Last modified time: 2020-02-20T20:20:22+01:00

import h5py
import numpy as np
from numpy import random

def generator(features, labels, batchSize=32, num=None, mode="all"):

    indices=np.arange(num)

    while True:

        np.random.shuffle(indices)
        for i in range(0, len(indices), batchSize):

            batch_indices = indices[i:i+batchSize]
            batch_indices.sort()

            if mode == "urban":
                by = labels[batch_indices, :10]
            else:
                by = labels[batch_indices, :]
            bx = features[batch_indices,:,:,:]

            yield (bx,by)

def generator_sampled(features, labels, batchSize=32, num=None, mode="all"):

    indices=np.arange(num)

    while True:

        np.random.shuffle(indices)
        # sample for each row of the labels matrix a random label using the row as probabilities
        labels_sampled = np.array([random.choice(range(labels.shape[1]), p=labels[i,:], size=1) for i in range(labels.shape[0])])
        labels_new = np.eye(labels.shape[1])[labels_sampled.reshape(-1,labels.shape[0])[0]]

        for i in range(0, len(indices), batchSize):

            batch_indices = indices[i:i+batchSize]
            batch_indices.sort()

            by = labels_new[batch_indices, :]
            bx = features[batch_indices,:,:,:]

            yield (bx,by, batch_indices)