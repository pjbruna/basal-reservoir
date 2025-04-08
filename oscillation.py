import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from reservoir import *
from functions import *


# Hyperparams

timesteps = 50
savefig = True


# Create oscillatory regime

train_data = [[0,1], [1,0]] * int(timesteps/2)
train_labels = ['1', '2'] * int(timesteps/2)


# Train model

model = SlimeMoldReservoir(input_nnodes=len(train_data[0]), nnodes=25, input_connectivity=0.5, leak=0.1, lrate_targ=0.01, lrate_wmat=0.1, targ_min=1)
results = model.run(train_data)

# plot_activations(results[0], 50, save=savefig)


# PCA

res_states = np.array(results[0])
res_states = res_states.T

pca = PCA(n_components=model.nnodes)
principal_components = pca.fit_transform(res_states)

print("Explained variance ratio:", pca.explained_variance_ratio_)

plot_pca_explainedvar(pca.explained_variance_ratio_, save=savefig)

pc1 = principal_components[:, 0]
pc2 = principal_components[:, 1]

timesteps = np.arange(res_states.shape[0])

plot_pca_space(pc1, pc2, timesteps, save=savefig)


# Cue trained model

cue = [[0,0], [0,0], [0,0], [0,0]]
cue_labels = ['NA', 'NA', 'NA', 'NA']

cue_results = model.echo(cue)

past_states = results[0]
cue_states = cue_results[0]

cue1_cor_matrix = correlate_states(past_states, cue_states)

plot_correlations(cue1_cor_matrix, train_labels, cue_labels, save=savefig)

print(cue_states.sum(axis=0)) # fading memory