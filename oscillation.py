import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from reservoir import *
from functions import *

### Run model ###

# Create training data

train_data = [[0,1], [1,0]] * 10
train_labels = ['1', '2'] * 10

# Train model

model = SlimeMoldReservoir(input_nnodes=len(train_data[0]), nnodes=10, input_connectivity=0.2, p_link=0.5, leak=0.1, lrate_targ=0.01, lrate_wmat=0.1, targ_min=1)
results = model.run(train_data)

# Plot activations

plot_activations(results[0], 20, save=True)


### PCA ###

res_states = np.array(results[0])[:,10:] # last 10 timesteps
res_states = res_states.T

pca = PCA(n_components=model.nnodes)
principal_components = pca.fit_transform(res_states)

print("Explained variance ratio:", pca.explained_variance_ratio_)

plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.grid()
plt.savefig('figures/pca_explained_var.png')
# plt.show()

pc1 = principal_components[:, 0]
pc2 = principal_components[:, 1]

timesteps = np.arange(res_states.shape[0])
colors = timesteps

plt.figure(figsize=(8, 6))
plt.plot(pc1, pc2, color='gray', alpha=1, linestyle='dotted', linewidth=0.8)
scatter = plt.scatter(pc1, pc2, c=colors, cmap='viridis', alpha=0.8)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.title('PCA Space')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(scatter, label='Time')
plt.grid()
plt.savefig('figures/pca.png')
# plt.show()


### Train linear readout ###

targets = np.array(train_labels)[10:].reshape(-1,1)

ridge = Ridge(alpha=1e-6)  # Regularization parameter alpha
ridge.fit(res_states, targets)

# W_out = ridge.coef_.T

predictions = ridge.predict(res_states)
print(predictions)

mse = np.mean((predictions - targets.astype(np.float64)) ** 2)
print("Mean Squared Error:", mse)

plt.figure(figsize=(8, 6))
plt.plot(predictions.flatten(), label="Prediction")
plt.plot(targets.flatten().astype(np.float64), label="Target", linestyle="dotted")
plt.title(f"Trained Readout (MSE: {mse})")
plt.xlabel("Timesteps")
plt.ylabel("Output")
plt.legend()
plt.grid()
plt.savefig('figures/trained_readout.png')
# plt.show()


### Cue trained model ###

state = 0 # select matrix for correlations: 0 - activations, 1 - reservoir weights

cue = [[0,0], [0,0], [0,0], [0,0]]
cue_labels = ['NA', 'NA', 'NA', 'NA']

cue_results = model.echo(cue)

print(cue_results[state].sum(axis=0)) # fading memory

cue1_cor_matrix = correlate_states(results[state], cue_results[state])
plot_correlations(cue1_cor_matrix, train_labels, cue_labels, save=True)