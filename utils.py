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

def dirichlet_variance(dirichlet_params):
    """
    Calculates the variance of a Dirichlet distribution.
    :param dirichlet_params: parameters of the Dirichlet distribution.
    :return: variance of the Dirichlet distribution.
    """
    alpha = dirichlet_params
    alpha0 = np.sum(alpha)
    return alpha * (alpha0 - alpha) / (alpha0 ** 2 * (alpha0 + 1))

def compute_calibration(true_labels, pred_labels, confidences, confidences_mat, num_bins=10):
    """Collects predictions into bins used to draw a reliability diagram.
    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins
    The true_labels, pred_labels, confidences arguments must be NumPy arrays;
    pred_labels and true_labels may contain numeric or string labels.
    For a multi-class model, the predicted label and confidence should be those
    of the highest scoring class.
    Returns a dictionary containing the following NumPy arrays:
        accuracies: the average accuracy for each bin
        confidences: the average confidence for each bin
        counts: the number of examples in each bin
        bins: the confidence thresholds for each bin
        avg_accuracy: the accuracy over the entire test set
        avg_confidence: the average confidence over the entire test set
        expected_calibration_error: a weighted average of all calibration gaps
        max_calibration_error: the largest calibration gap across all bins
    """
    assert (len(confidences) == len(pred_labels))
    assert (len(confidences) == len(true_labels))
    assert (num_bins > 0)

    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=np.float)
    bin_confidences = np.zeros(num_bins, dtype=np.float)
    bin_counts = np.zeros(num_bins, dtype=np.int)
    bin_class_counts = np.zeros((num_bins, 16), dtype=np.int)
    bin_class_accuracies = np.zeros((num_bins, 16), dtype=np.float)
    bin_class_confidences = np.zeros((num_bins, 16), dtype=np.float)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)
            for k in range(16):
                bin_class_counts[b, k] = np.sum(true_labels[selected] == k + 1)
                if bin_class_counts[b, k] > 0:
                    bin_class_accuracies[b, k] = np.mean((true_labels[selected] == pred_labels[selected])[true_labels[selected] == k + 1])
                    bin_class_confidences[b, k] = np.mean(confidences_mat[selected, k][true_labels[selected] == k + 1])

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)

    class_gaps = np.abs(bin_class_accuracies - bin_class_confidences)
    class_sces = np.mean(class_gaps*bin_class_counts, axis=0)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)
    # We now have K=16 classes, so we need to adjust the static calibration error
    sce = 1/16 * np.sum(class_gaps * bin_class_counts) / np.sum(bin_class_counts)

    return {"accuracies": bin_accuracies,
            "confidences": bin_confidences,
            "counts": bin_counts,
            "bins": bins,
            "avg_accuracy": avg_acc,
            "avg_confidence": avg_conf,
            "expected_calibration_error": ece,
            "max_calibration_error": mce,
            "static_calibration_error": sce,
            "class_sces": class_sces}