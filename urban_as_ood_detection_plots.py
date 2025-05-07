import numpy as np
import h5py
import os
import yaml
import json
from pathlib import Path
import model_without_softmax
import model_with_softmax

from utils import compute_calibration

import gc

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve

import tensorflow as tf

from scipy.special import psi
import numpy as np
import tensorflow as tf

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

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

# Equalize number of samples in test_patches_id and test_patches_ood
if test_patches_id.shape[0] > test_patches_ood.shape[0]:
    test_patches_id = test_patches_id[:test_patches_ood.shape[0], :, :, :]
elif test_patches_id.shape[0] < test_patches_ood.shape[0]:
    test_patches_ood = test_patches_ood[:test_patches_id.shape[0], :, :, :]
else:
    pass
# construct numpy array with 0 for ID and 1 for OOD samples
targets_id = np.zeros(test_patches_id.shape[0])
targets_ood = np.ones(test_patches_ood.shape[0])
targets = np.concatenate((targets_id, targets_ood), axis=0)

# Choose model

mode = "dirichlet"

path = os.getcwd()

with open("configs/model_settings.yaml", 'r') as fp:
    setting_dict = yaml.load(fp, Loader=yaml.FullLoader)

# Set hyperparameters accordingly
seed = 0
setting_dict["Seed"] = seed
setting_dict["Data"]["mode"] = mode
batchSize = setting_dict["Data"]["train_batch_size"]
lrate = setting_dict["Optimization"]["lr"]
# Derive model checkpoint filename>
res_ckpt_filepath = Path(path, "results", f"Sen2LCZ_bs_{batchSize}_lr_{lrate}_seed_{seed}_weights_best_{mode}_veg_as_id_short.hdf5")

if mode == "one-hot" or mode == "distributional":
    model = model_with_softmax.sen2LCZ_drop(depth=17,
                                            dropRate=setting_dict["Data"]["dropout"],
                                            fusion=setting_dict["Data"]["fusion"],
                                            num_classes=7)
elif mode == "dirichlet" or mode == "Dirichlet_embedding" or mode == "MSE_embedding" or mode == "Mahala_embedding":
    model = model_without_softmax.sen2LCZ_drop(depth=17,
                                               dropRate=setting_dict["Data"]["dropout"],
                                               fusion=setting_dict["Data"]["fusion"],
                                               num_classes=7)
print("Model configured")

model.load_weights(res_ckpt_filepath, by_name=False)
# Store predictions + corresponding confidence
preds_id = model.predict(test_patches_id, batch_size=setting_dict["Data"]["test_batch_size"])
preds_ood = model.predict(test_patches_ood, batch_size=setting_dict["Data"]["test_batch_size"])
# Transform raw predictions into pseudo softmax probabilities
if mode == "dirichlet" or mode == "Dirichlet_embedding" or mode == "MSE_embedding" or mode == "Mahala_embedding":
    # For Dirichlet approaches, scale up logits by temperature = 3 (since we scaled down during training)
    if mode == "dirichlet" or mode == "Dirichlet_embedding":
        alphas_id = np.exp(preds_id * 3)
        alphas_ood = np.exp(preds_ood * 3)
    else:
        alphas_id = np.exp(preds_id)
        alphas_ood = np.exp(preds_ood)
    # For all cases, apply softmax transformation
    probs_id = alphas_id / np.sum(alphas_id, axis=-1, keepdims=True)
    probs_ood = alphas_ood / np.sum(alphas_ood, axis=-1, keepdims=True)
if mode == "one-hot" or mode == "distributional":
    probs_id = preds_id
    probs_ood = preds_ood
    auroc_dsm = "NA"
    aupr_dsm = "NA"
    auroc_exp_ent = "NA"
    aupr_exp_ent = "NA"
    auroc_evidence = "NA"
    aupr_evidence = "NA"
    auroc_distr_unc = "NA"
    aupr_distr_unc = "NA"
msp_id = 1 - np.max(probs_id, axis=-1)
msp_ood = 1 - np.max(probs_ood, axis=-1)
pred_ent_id = -np.sum(probs_id * np.log(probs_id), axis=-1)
pred_ent_ood = -np.sum(probs_ood * np.log(probs_ood), axis=-1)
auroc_msp = roc_auc_score(targets, np.concatenate((msp_id, msp_ood), axis=0))
auroc_pred_ent = roc_auc_score(targets, np.concatenate((pred_ent_id, pred_ent_ood), axis=0))
aupr_msp = auc(*precision_recall_curve(targets, np.concatenate((msp_id, msp_ood), axis=0))[-2:-4:-1])
aupr_pred_ent = auc(*precision_recall_curve(targets, np.concatenate((pred_ent_id, pred_ent_ood), axis=0))[-2:-4:-1])
if mode == "dirichlet" or mode == "Dirichlet_embedding" or mode == "MSE_embedding" or mode == "Mahala_embedding":
    dsm_id = 7 / (np.sum(alphas_id, axis=-1, keepdims=True) + 7)
    dsm_ood =7 / (np.sum(alphas_ood, axis=-1, keepdims=True) + 7)
    exp_ent_id = np.sum(probs_id * (psi(alphas_id + 1) - psi(np.sum(alphas_id, axis=-1, keepdims=True))), axis=-1)
    exp_ent_ood = np.sum(probs_ood * (psi(alphas_ood + 1) - psi(np.sum(alphas_ood, axis=-1, keepdims=True))), axis=-1)
    evidence_id = np.sum(alphas_id, axis=-1)
    evidence_ood = np.sum(alphas_ood, axis=-1)
    distr_unc_id = -np.sum(
        probs_id * (np.log(probs_id) - psi(alphas_id + 1) + psi(np.sum(alphas_id + 1, axis=-1, keepdims=True))),
        axis=-1)
    distr_unc_ood = -np.sum(
        probs_ood * (np.log(probs_ood) - psi(alphas_ood + 1) + psi(np.sum(alphas_ood + 1, axis=-1, keepdims=True))),
        axis=-1)
    auroc_dsm = roc_auc_score(targets, np.concatenate((dsm_id, dsm_ood), axis=0))
    auroc_exp_ent = roc_auc_score(targets, np.concatenate((exp_ent_id, exp_ent_ood), axis=0))
    auroc_evidence = roc_auc_score(targets, np.concatenate((evidence_id, evidence_ood), axis=0))
    auroc_distr_unc = roc_auc_score(targets, np.concatenate((distr_unc_id, distr_unc_ood), axis=0))
    aupr_dsm = auc(*precision_recall_curve(targets, np.concatenate((dsm_id, dsm_ood), axis=0))[-2:-4:-1])
    aupr_exp_ent = auc(*precision_recall_curve(targets, np.concatenate((exp_ent_id, exp_ent_ood), axis=0))[-2:-4:-1])
    aupr_evidence = auc(*precision_recall_curve(targets, np.concatenate((evidence_id, evidence_ood), axis=0))[-2:-4:-1])
    aupr_distr_unc = auc(
        *precision_recall_curve(targets, np.concatenate((distr_unc_id, distr_unc_ood), axis=0))[-2:-4:-1])

uncertainties_ood = msp_ood
uncertainties_ood = np.expand_dims(uncertainties_ood, axis=1)
uncertainties_id = msp_id
uncertainties_id = np.expand_dims(uncertainties_id, axis=1)

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

unc_id_ood = np.hstack((uncertainties_id, uncertainties_ood))
unc_id_ood = trunc(unc_id_ood, 2)

unc_id_ood = pd.DataFrame(unc_id_ood, columns=['ID uncertainty', 'OoD uncertainty'])

bins = np.linspace(0,1,100)

# ROC curve + AUROC

scores = np.concatenate((uncertainties_id, uncertainties_ood))

false_positive_rate, true_positive_rate, threshold = roc_curve(targets, scores)

auroc = roc_auc_score(targets, scores)

# Uncertainty barplot


plt.hist(unc_id_ood, bins, label=['ID uncertainty', 'OoD uncertainty'])
plt.legend(loc='upper right')
plt.xlabel('1 - Maximum Softmax Probability')
plt.ylabel('Frequency')
plt.show()

matplotlib.rcParams.update({'font.size': 22})

plt.subplots(1, figsize=(10,10))
#plt.title('Receiver Operating Characteristic: OoD = FashionMNIST')
plt.plot(false_positive_rate, true_positive_rate, label="ROC curve (area = %0.2f)" % auroc)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc="lower right")
plt.show()