import h5py
import numpy as np
from operator import add
import pandas as pd
import tensorflow as tf

def process_city(city):
    name_tmp = "E:/Dateien/LCZ_Votes/" + city + ".h5"
    h5_tmp = h5py.File(name_tmp,'r')
    votes_mat = pd.DataFrame(np.array(h5_tmp['label']))
    city_mat = np.array([city] * len(h5_tmp['label']))
    votes_mat['City'] = city_mat
    return votes_mat

def to_one_hot(vote_mat, labels):
    one_hot_encoded_mat = list()
    for i in range(len(vote_mat)):
        one_hot_encoded = list()
        vote_vec = vote_mat.iloc[i,:].values
        for value in vote_vec:
            one = [0 for _ in range(len(labels))]
            one[value - 1] = 1
            if one_hot_encoded:
                one_hot_encoded = list(map(add, one_hot_encoded, one))
            else: # initialize one_hot_encoded if list is yet empty
                one_hot_encoded = one
        if i == 0:
            one_hot_encoded_mat = np.asarray(one_hot_encoded)
        else:
            one_hot_encoded_mat = np.vstack((one_hot_encoded_mat, np.asarray(one_hot_encoded)))
    return(one_hot_encoded_mat)

def concatenate_cities(cities_list):
    concatenated_mat = np.array([])
    for city in cities_list:
        name_tmp = "E:/Dateien/LCZ_Votes/" + city + ".h5"
        h5_tmp = h5py.File(name_tmp,'r')
        if concatenated_mat.size == 0:
            concatenated_mat = np.array(h5_tmp['label'])
        else:
            concatenated_mat = np.vstack((concatenated_mat, np.array(h5_tmp['label'])))
    return(concatenated_mat)

def concatenate_cities_patches(cities_list):
    concatenated_mat = np.array([])
    for city in cities_list:
        name_tmp = "E:/Dateien/LCZ_Votes/" + city + ".h5"
        h5_tmp = h5py.File(name_tmp,'r')
        # If yet empty, initialize matrix with first city file
        if concatenated_mat.size == 0:
            concatenated_mat = np.array(h5_tmp['sen2'])
        # Otherwise: append existing matrix with new city file
        else:
            concatenated_mat = np.vstack((concatenated_mat, np.array(h5_tmp['sen2'])))
    return(concatenated_mat)

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

def ECE(predictions, labels, bins):
    """
    Calculates the Expected Calibration Error of a set of predictions.
    See https://arxiv.org/abs/1706.04599 for details.
    :param predictions: matrix of softmax predictions, each row sums to 1.
    :param labels: matrix of one-hot encoded labels. labels[i, j] = 1 iff i == j.
    :param bins: number of confidence interval bins.
    :return: Expected Calibration Error.
    """
    softmaxes = predictions
    confidences = np.max(softmaxes, axis=1)
    predictions = np.argmax(softmaxes, axis=1)
    accuracies = predictions == labels

    ece = 0
    bin_size = 1 / bins
    for bin_i in range(bins):
        # Calculated |confidence - accuracy| in each bin
        in_bin = (confidences >= bin_i * bin_size) & (confidences < (bin_i + 1) * bin_size)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece