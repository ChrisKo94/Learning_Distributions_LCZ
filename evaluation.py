import numpy as np
import h5py
import os
import yaml
import json
from pathlib import Path
import pandas as pd
import model_without_softmax
import model_with_softmax
import pandas as pd

from utils import compute_calibration

import gc

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score

import tensorflow as tf

## Set path ##
path = os.getcwd()
results_dir = Path(path, 'results')
results_dir.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

data_embedding = h5py.File('E:/Dateien/LCZ_Votes/embedding_data.h5', 'r')
patches = np.array(data_embedding.get("x"))
labels = np.array(data_embedding.get("y_one_hot"))
embeddings = np.array(data_embedding.get("y"))
label_distributions = np.array(data_embedding.get("y_distributional"))

#Train Val Test Split
shuffled_indices = np.random.permutation(patches.shape[0])

patches = patches[shuffled_indices,:,:,:]
train_patches, val_patches, patches_test = np.split(patches, [int(.6*len(patches)), int(.8*len(patches))])

labels = labels[shuffled_indices, :]
train_labels, val_labels, labels_test = np.split(labels, [int(.6 * len(labels)), int(.8 * len(labels))])

embeddings = embeddings[shuffled_indices, :]
train_embeddings, val_embeddings, embeddings_test = np.split(embeddings, [int(.6 * len(embeddings)), int(.8 * len(embeddings))])

embedding_distributions_test = np.exp(embeddings_test) / np.sum(np.exp(embeddings_test), axis=-1, keepdims=True)

train_label_distributions, val_label_distributions, label_distributions_test = np.split(label_distributions, [int(.6 * len(label_distributions)), int(.8 * len(label_distributions))])

## Save results to dataframe
results = pd.DataFrame()
sce_values = np.empty((0, 16), float)
kl_to_embedding_values = np.empty((0, 16), float)
kl_to_embedding_stds = np.empty((0, 16), float)
kl_to_empirical_distr_values = np.empty((0, 16), float)

## Model prediction ##

def evaluation(res_ckpt_filepath):

    ## Model settings
    if mode == "one-hot" or mode == "distributional" or mode == "sampled_one-hot":
        model = model_with_softmax.sen2LCZ_drop(depth=17,
                                               dropRate=setting_dict["Data"]["dropout"],
                                               fusion=setting_dict["Data"]["fusion"],
                                               num_classes=setting_dict["Data"]["num_classes"])
    elif mode == "dirichlet" or mode == "Dirichlet_embedding" or mode == "MSE_embedding" or mode == "Mahala_embedding":
        model = model_without_softmax.sen2LCZ_drop(depth=17,
                                               dropRate=setting_dict["Data"]["dropout"],
                                               fusion=setting_dict["Data"]["fusion"],
                                               num_classes=setting_dict["Data"]["num_classes"])
    print("Model configured")

    model.load_weights(res_ckpt_filepath, by_name=False)
    # Store predictions + corresponding confidence
    y_pre_prob = model.predict(patches_test, batch_size = setting_dict["Data"]["test_batch_size"])
    # Transform raw predictions into pseudo softmax probabilities
    if mode == "dirichlet" or mode == "Dirichlet_embedding" or mode == "MSE_embedding" or mode == "Mahala_embedding":
        # For Dirichlet approaches, scale up logits by temperature = 3 (since we scaled down during training)
        if mode == "dirichlet" or mode == "Dirichlet_embedding":
            y_pre_prob = y_pre_prob*3
        # For all cases, apply softmax transformation
        y_pre_prob = np.exp(y_pre_prob) / np.sum(np.exp(y_pre_prob), axis=-1, keepdims=True)
    y_pre = y_pre_prob.argmax(axis=-1)+1
    confidence = y_pre_prob[np.arange(y_pre_prob.shape[0]), (y_pre - 1).tolist()]
    y_testV = labels_test.argmax(axis=-1)+1
    # Compute performance metrics
    classRep = classification_report(y_testV, y_pre, digits=4, output_dict=True)
    oa = accuracy_score(y_testV, y_pre)
    macro_avg = classRep["macro avg"]["precision"]
    weighted_avg = classRep["weighted avg"]["precision"]
    cohKappa = cohen_kappa_score(y_testV, y_pre)

    # New calibration metrics
    epsilon = 1e-4
    KL_to_embedding_matrix = np.sum(embedding_distributions_test * (np.log(embedding_distributions_test+epsilon) - np.log(y_pre_prob+epsilon)), axis=-1)
    # KL to embedding categorized by unique values of y_testV
    KL_to_embedding_by_class = np.array([np.mean(KL_to_embedding_matrix[y_testV == i]) for i in np.arange(1,17)])
    # Overall KL to embedding
    KL_to_embedding = np.mean(KL_to_embedding_matrix)
    # Standard deviation of KL to embedding (NEW, todo: check if useful)
    KL_to_embedding_std = np.std(KL_to_embedding_matrix)
    # KL to empirical distribution
    KL_to_empirical_distr_matrix= np.sum(label_distributions_test * (np.log(label_distributions_test+epsilon) - np.log(y_pre_prob+epsilon)), axis=-1)
    # KL to empirical distribution categorized by unique values of y_testV
    KL_to_empirical_distr_by_class = np.array([np.mean(KL_to_empirical_distr_matrix[y_testV == i]) for i in np.arange(1,17)])
    KL_to_empirical_distr = np.mean(KL_to_empirical_distr_matrix)

    ece = compute_calibration(y_testV,y_pre,confidence,y_pre_prob,num_bins=setting_dict["Calibration"]["n_bins"])['expected_calibration_error']
    mce = compute_calibration(y_testV,y_pre,confidence,y_pre_prob,num_bins=setting_dict["Calibration"]["n_bins"])['max_calibration_error']
    sce = compute_calibration(y_testV,y_pre,confidence,y_pre_prob,num_bins=setting_dict["Calibration"]["n_bins"])['static_calibration_error']
    class_gaps = compute_calibration(y_testV,y_pre,confidence,y_pre_prob,num_bins=setting_dict["Calibration"]["n_bins"])['class_sces']

    # Store results
    res = {
        'oa': float(oa),
        'maa': macro_avg,
        'waa': weighted_avg,
        'kappa': float(cohKappa),
        'ece': ece,
        'mce': mce,
        'sce': sce,
        'KL_to_embedding': KL_to_embedding,
        'KL_to_empirical_distr': KL_to_empirical_distr
    }



    # Clear memory

    del y_pre_prob
    del y_pre
    del confidence
    del y_testV

    gc.collect()

    # Create results file
    output_path_res = Path(res_ckpt_filepath.parent, f"{res_ckpt_filepath.stem}_results.json")
    output_path_res.parent.mkdir(parents=True, exist_ok=True)
    # Write results to disk
    with open(output_path_res, 'w') as f:
        json.dump(res, f)
        print("Starting Evaluating:")
        print(res)

    return res, class_gaps, KL_to_embedding_by_class, KL_to_empirical_distr_by_class

## Load settings dictionary ##

with open("configs/model_settings.yaml", 'r') as fp:
    setting_dict = yaml.load(fp, Loader=yaml.FullLoader)

## Evaluate models ##

if __name__ == "__main__":
    for mode in ["one-hot", "distributional", "sampled_one-hot", "dirichlet", "Dirichlet_embedding", "MSE_embedding", "Mahala_embedding"]: #
        for seed in range(1):
            # Set hyperparameters accordingly
            setting_dict["Seed"] = seed
            setting_dict["Data"]["mode"] = mode
            batchSize = setting_dict["Data"]["train_batch_size"]
            lrate = setting_dict["Optimization"]["lr"]
            # Derive model checkpoint filename
            res_ckpt_filepath = Path(path, "results", f"Sen2LCZ_bs_{batchSize}_lr_{lrate}_seed_{seed}_weights_best_{mode}_random_split_30.hdf5")
            # Start evaluation
            res, class_sces, class_kl_to_embedding, class_kl_to_empirical_distr = evaluation(res_ckpt_filepath)
            # Store results in overall results matrix
            results = results.append(res, ignore_index=True)
            # return class_gaps
            sce_values = np.append(sce_values, class_sces.reshape(1,-1), axis=0)
            # return KL to embedding
            kl_to_embedding_values = np.append(kl_to_embedding_values, class_kl_to_embedding.reshape(1,-1), axis=0)
            # return KL to empirical distribution
            kl_to_empirical_distr_values = np.append(kl_to_empirical_distr_values, class_kl_to_empirical_distr.reshape(1,-1), axis=0)
# Write ALL results to disk
results.to_csv(Path(path,"results","0.0002_results_random_split_30.csv"))
# Transform sce_values into dataframe with mode as index
sce_values = pd.DataFrame(sce_values, index=["One-Hot", "Distr.", "Sampled One-Hot", "Simple Dirichlet", "KL Embedd.", "MSE Embedd.", "MD Embedd."])
# Write sce_values to disk
sce_values.to_csv(Path(path,"results","0.0002_sce_values_random_split_20.csv"))
# Transform kl_to_embedding_values into dataframe with mode as index
kl_to_embedding_values = pd.DataFrame(kl_to_embedding_values, index=["One-Hot", "Distr.", "Sampled One-Hot", "Simple Dirichlet", "KL Embedd.", "MSE Embedd.", "MD Embedd."])
# Write kl_to_embedding_values to disk
kl_to_embedding_values.to_csv(Path(path,"results","0.0002_kl_to_embedding_values_random_split_20.csv"))
# Transform kl_to_empirical_distr_values into dataframe with mode as index
kl_to_empirical_distr_values = pd.DataFrame(kl_to_empirical_distr_values, index=["One-Hot", "Distr.", "Sampled One-Hot", "Simple Dirichlet", "KL Embedd.", "MSE Embedd.", "MD Embedd."])
# Write kl_to_empirical_distr_values to disk
kl_to_empirical_distr_values.to_csv(Path(path,"results","0.0002_kl_to_empirical_distr_values_random_split_20.csv"))
