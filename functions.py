import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm

## HELPER FUNCTIONS ##

# Correlation matrix between spike states

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

# Plot state correlations

def plot_correlations(r_mat, train_labels, cue_labels):
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

  plt.show()

# Plot weights over time

def plot_weights(weights, timesteps, rows, cols):

  fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
  axes = axes.flatten()

  for t in range(timesteps):
    matrix = np.array(weights.iloc[:,-(t+1)]).reshape((100,100))

    im = axes[t].imshow(matrix, cmap='viridis', interpolation='nearest')
    fig.colorbar(im, ax=axes[t])
    axes[t].set_title(f"Time Step -{timesteps-t}")

  plt.tight_layout()
  plt.show()

def plot_activations(acts, timesteps):
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
  plt.show()

