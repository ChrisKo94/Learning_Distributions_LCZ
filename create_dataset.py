import numpy as np
import pandas as pd
import h5py

from utils import *

########################################################################################################################
################################ Data Generation (matching with Katharina's code) ######################################
########################################################################################################################

city_list = ['berlin', 'cologne', 'london', 'madrid', 'milan', 'munich', 'paris', 'rome', 'zurich']

# Define labels
labels = np.arange(1,18)

cities_frames = [process_city(city) for city in city_list]
cities_votes_named = pd.concat(cities_frames)

cities_votes = pd.DataFrame(concatenate_cities(city_list).astype(int))

# to one hot

cities_one_hot = pd.DataFrame(to_one_hot(cities_votes, labels))
cities_one_hot_named = cities_one_hot.copy()
cities_one_hot_named['City'] = np.array(cities_votes_named['City'])

# Extract patches

cities_patches = concatenate_cities_patches(city_list)
indeces_out = np.array(cities_one_hot[cities_one_hot[6] != 0].index)
cities_patches = np.delete(cities_patches, indeces_out, 0)

# delete instances with vote for class 7

# But before that, add vote counts of class 7 to class 3
cities_one_hot[3] = cities_one_hot[3] + cities_one_hot[6]

cities_one_hot_16 = cities_one_hot.drop(cities_one_hot[cities_one_hot[6] != 0].index)
cities_one_hot_16 = cities_one_hot_16.drop(columns=[6])

katharina_embeddings = pd.read_csv("E:/Downloads/Archiv/df_z_full_all_img.csv")

# 2. unique patterns with frequency for model ----------------------

vote_patterns = np.unique(cities_one_hot_16, axis=0, return_inverse=True, return_counts=True)

# 3. import csv of embeddings from R -------------------------------

z_hat = pd.read_csv('E:/Downloads/Archiv/z_full.csv', ) #Todo: Change path to project dir

z_hat = np.array(z_hat.drop(z_hat.columns[0], axis=1))

mapping = vote_patterns[1]
z_all_images = []
for m in mapping:
    z_all_images.append(z_hat[m])

z_all_images = np.array(z_all_images)

# 4. add distributional & one-hot labels  --------------------------

cities_one_hot_16_majority = np.argmax(np.array(cities_one_hot_16), axis=1)

one_hot_encoded_array = np.zeros((cities_one_hot_16_majority.size, cities_one_hot_16_majority.max()+1), dtype=int)

#replacing 0 with a 1 at the index of the original array
one_hot_encoded_array[np.arange(cities_one_hot_16_majority.size),cities_one_hot_16_majority] = 1

distributional_array = np.array(cities_one_hot_16) / np.array(cities_one_hot_16).sum(axis=1)[:,None]

# 5. save data -----------------------------------------------------

data_h5 = h5py.File('E:/Dateien/LCZ_Votes/embedding_data.h5', 'w')
data_h5.create_dataset('x', data=cities_patches)
data_h5.create_dataset('y', data=z_all_images)
data_h5.create_dataset('y_one_hot', data=one_hot_encoded_array)
data_h5.create_dataset('y_distributional', data=distributional_array)
data_h5.close()
