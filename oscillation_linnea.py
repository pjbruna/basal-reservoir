import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
from reservoir_linnea import SlimeMoldReservoir  # or your actual module name
from functions import plot_pca_explainedvar, plot_pca_space, plot_correlations, correlate_states

# Hyperparams
timesteps = 50
savefig = False

# Create oscillatory regime
train_data = [[1], [0]] * (timesteps // 2)
train_labels = ['ON', 'OFF'] * (timesteps // 2)

# Create and train the model
model = SlimeMoldReservoir(
    input_nnodes=len(train_data[0]),
    nnodes=36,
    input_connectivity=1.0,
    leak=0.1,
    p_link=0.2,
    lrate_targ=0.1,
    lrate_wmat=0.1,
    targ_min=1.0,
    network_type='hexagonal',  # you can also try 'grid', 'hexagonal', or 'random'
    seed=None,
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
cue = [[0], [0]] * 2
cue_labels = ['NA', 'NA'] * 2

cue_results = model.echo(cue)

past_states = results[0]
cue_states = cue_results[0]

cue1_cor_matrix = correlate_states(past_states, cue_states)
plot_correlations(cue1_cor_matrix, train_labels, cue_labels, save=savefig)

print(cue_states.sum(axis=0))  # Fading memory


# model.animate_weight_and_activation(save_path="weight_evolution.mp4")



# Fading memory PCA

# cue_states = np.array(cue_results[0]).T
# 
# cue_principal_components = pca.transform(cue_states)
# 
# tail_pc1 = principal_components[-12:, 0]
# tail_pc2 = principal_components[-12:, 1]
# 
# cue_pc1 = cue_principal_components[:, 0]
# cue_pc2 = cue_principal_components[:, 1]
# 
# pc1 = np.concatenate((tail_pc1, cue_pc1))
# pc2 = np.concatenate((tail_pc2, cue_pc2))
# 
# cue_timesteps_arr = np.arange(len(pc1))
# colors = cue_timesteps_arr
# 
# plt.figure(figsize=(8, 6))
# plt.plot(pc1, pc2, color='gray', alpha=1, linestyle='dotted', linewidth=0.8)
# scatter = plt.scatter(pc1[:-4], pc2[:-4], c=colors[:-4], cmap='viridis', alpha=0.8)
# plt.scatter(pc1[-4:], pc2[-4:], color='red', alpha=0.8, label='Cue points')
# plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
# plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
# plt.title('PCA Space')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.colorbar(scatter, label='Time')
# plt.grid()
# plt.show()

# if save==True:
#   timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#   plt.savefig(f'figures/pca_{timestamp}.png')
# else:
#   plt.show()