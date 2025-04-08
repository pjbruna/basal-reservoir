import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from reservoir_linnea import SlimeMoldReservoir  # or your actual module name
from functions import plot_pca_explainedvar, plot_pca_space, plot_correlations, correlate_states

# Hyperparams
timesteps = 50
savefig = True

# Create oscillatory regime
train_data = [[0, 1], [1, 0]] * (timesteps // 2)
train_labels = ['1', '2'] * (timesteps // 2)

# Create and train the model
model = SlimeMoldReservoir(
    input_nnodes=len(train_data[0]),
    nnodes=36,
    input_connectivity=1.0,
    leak=0.01,
    p_link=0.2,
    lrate_targ=0.01,
    lrate_wmat=0.1,
    targ_min=1.0,
    network_type='hexagonal',  # you can also try 'grid', 'hexagonal', or 'random'
    seed=42,
    decay_rate=0.00
)

# Visualize initial network (optional)
model.plot_initial_network(show_inputs=True)

model.print_network_info()


# Run training
results = model.run(train_data)

# PCA
res_states = np.array(results[0]).T  # shape: (timesteps, nnodes)
pca = PCA(n_components=model.nnodes)
principal_components = pca.fit_transform(res_states)

print("Explained variance ratio:", pca.explained_variance_ratio_)

plot_pca_explainedvar(pca.explained_variance_ratio_, save=savefig)

pc1 = principal_components[:, 0]
pc2 = principal_components[:, 1]
timesteps_arr = np.arange(res_states.shape[0])

plot_pca_space(pc1, pc2, timesteps_arr, save=savefig)

# Cue trained model
cue = [[0, 0]] * 4
cue_labels = ['NA'] * 4

cue_results = model.echo(cue)

past_states = results[0]
cue_states = cue_results[0]

cue1_cor_matrix = correlate_states(past_states, cue_states)
plot_correlations(cue1_cor_matrix, train_labels, cue_labels, save=savefig)

print(cue_states.sum(axis=0))  # Fading memory


model.animate_weight_and_activation(save_path="weight_evolution.mp4")

