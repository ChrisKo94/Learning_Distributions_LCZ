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

data_embedding = h5py.File('E:/Dateien/LCZ_Votes/embedding_data.h5', 'r')
patches = np.array(data_embedding.get("x"))
labels = np.array(data_embedding.get("y_one_hot"))
embeddings = np.array(data_embedding.get("y"))
label_distributions = np.array(data_embedding.get("y_distributional"))

patches_test = patches[cities_geo == 0]
labels_test = labels[cities_geo == 0]
embeddings_test = embeddings[cities_geo == 0]
embedding_distributions_test = np.exp(embeddings_test) / np.sum(np.exp(embeddings_test), axis=-1, keepdims=True)
label_distributions_test = label_distributions[cities_geo == 0]

## Save results to dataframe
results = pd.DataFrame()
sce_values = np.empty((0, 16), float)
kl_to_embedding_values = np.empty((0, 16), float)
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
            res_ckpt_filepath = Path(path, "results", f"Sen2LCZ_bs_{batchSize}_lr_{lrate}_seed_{seed}_weights_best_{mode}.hdf5")
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
results.to_csv(Path(path,"results","0.0002_results.csv"))
# Transform sce_values into dataframe with mode as index
sce_values = pd.DataFrame(sce_values, index=["one-hot", "distributional", "sampled_one-hot", "dirichlet", "Dirichlet_emb.", "MSE_emb.", "Mahala_emb."])
# Write sce_values to disk
sce_values.to_csv(Path(path,"results","0.0002_sce_values.csv"))
# Transform kl_to_embedding_values into dataframe with mode as index
kl_to_embedding_values = pd.DataFrame(kl_to_embedding_values, index=["one-hot", "distributional", "sampled_one-hot", "dirichlet", "Dirichlet_embedding", "MSE_embedding", "Mahala_embedding"])
# Write kl_to_embedding_values to disk
kl_to_embedding_values.to_csv(Path(path,"results","0.0002_kl_to_embedding_values.csv"))
# Transform kl_to_empirical_distr_values into dataframe with mode as index
kl_to_empirical_distr_values = pd.DataFrame(kl_to_empirical_distr_values, index=["one-hot", "distributional", "sampled_one-hot", "dirichlet", "Dirichlet_embedding", "MSE_embedding", "Mahala_embedding"])
# Write kl_to_empirical_distr_values to disk
kl_to_empirical_distr_values.to_csv(Path(path,"results","0.0002_kl_to_empirical_distr_values.csv"))
