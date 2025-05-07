import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

num_classes = 16
palette = np.hstack([np.array([sns.color_palette("Set1")]),
                     np.array([sns.color_palette("Set2")])[:,[0,2,4,6],:],
                     np.array([sns.color_palette("Set3")])[:,[4,9],:],
                     np.array([sns.color_palette("Greys")])[:,[5],:],
                     np.array([sns.color_palette("OrRd")])[:,[5],:]])[:,:num_classes,:]

index = ["1", "2", "3", "4", "5", "6", "8", "9", "10", "A", "B", "C", "D", "E", "F", "G"]
sces = pd.read_csv("results/0.0002_sce_values_random_split_30.csv", index_col=0)
sces = pd.DataFrame(sces.values, columns=index, index=sces.index)

# Plot SCE values for each class for each model in a bar plot with different colors for each class using the palette

sns.set_palette(palette[0])

ax = sces.plot.bar(rot=0, figsize=(10, 5))
plt.legend(title="Class")
plt.xlabel("Model")
plt.ylabel("SCE Class Contribution")
# Shift legend to the right
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

