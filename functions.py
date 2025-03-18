import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
from datetime import datetime

## HELPER FUNCTIONS ##


# Correlation matrix between states

def correlate_states(train_data, cue_data):
  plot_data = np.array(train_data.iloc[:,-12:])
  plot_data = np.hstack((plot_data, np.array(cue_data)))
  cor_matrix = np.zeros((16,16))
  for n in range(16):
    x = plot_data[:,n]
    for r in range(16):
      if (r >= n):  
        y = plot_data[:,r]
        res = np.around(scipy.stats.pearsonr(x, y)[0],2)
        cor_matrix[r,n]=res
      else:
        cor_matrix[r,n] = np.nan

  return cor_matrix


# Plot correlations

def plot_correlations(r_mat, train_labels, cue_labels, save=True):
  cmap = cm.get_cmap('RdYlBu', 7)

  fig, ax = plt.subplots()
  im = ax.imshow(r_mat, cmap=cmap)

  ax.set_xticks(np.arange(16))
  ax.set_yticks(np.arange(16))

  ax.set_xticklabels(np.append(np.array(train_labels[-12:]), np.array(cue_labels)).tolist())
  ax.set_yticklabels(np.append(np.array(train_labels[-12:]), np.array(cue_labels)).tolist())

  plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

  for i in range(16):
    for j in range(16):
      if j > i:
        continue
      text = ax.text(j, i, r_mat[i, j], ha="center", va="center", color="black")

  plt.colorbar(im, ax=ax)
  im.set_clim(-1, 1)
  fig.tight_layout()
  fig.set_size_inches(13.75, 10)

  if save==True:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'figures/corr_mat_{timestamp}.png')
  else:
    plt.show()


# Plot node activations over time

def plot_activations(acts, timesteps, save=True):
  data = np.array(acts)

  timesteps = min(timesteps, data.shape[1])
  data = data[:,-timesteps:]
  time = np.arange(data.shape[1] - timesteps, data.shape[1])

  plt.figure(figsize=(12, 8))
  for i in range(data.shape[0]):
      plt.plot(time, data[i, :], label=f'Node {i+1}', alpha=0.5)
  
  # Labeling the plot
  plt.xlabel('Timesteps')
  plt.ylabel('Activation')
  plt.title(f'Node Values Over Last {timesteps} Timesteps')
  plt.legend(loc='upper right')

  if save==True:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'figures/plot_acts_{timestamp}.png')
  else:
    plt.show()


# PCA on reservoir

def plot_pca_explainedvar(explained_variance, save=True):
  plt.figure(figsize=(8, 6))
  plt.plot(np.cumsum(explained_variance), marker='o')
  plt.xlabel('Number of Components')
  plt.ylabel('Cumulative Explained Variance')
  plt.title('Explained Variance by PCA Components')
  plt.grid()

  if save==True:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'figures/pca_explained_var_{timestamp}.png')
  else:
    plt.show()


def plot_pca_space(pc1, pc2, timesteps, save=True):
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

  if save==True:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'figures/pca_{timestamp}.png')
  else:
    plt.show()