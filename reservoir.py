import numpy as np
import pandas as pd
import random

## MODEL ##

class ResonanceNetwork:
  def __init__(self, input_nnodes=None, nnodes=None, p_link=None, leak=None, lrate_targ=None, lrate_wmat=None, targ_min=None):
    # Model hyperparameters
    self.nnodes = nnodes
    self.p_link = p_link
    self.leak = leak
    self.lrate_targ = lrate_targ
    self.lrate_wmat = lrate_wmat
    self.targ_min = targ_min

    if input_nnodes is not None:
      # Creat input layer and weights
      self.input_wmat = np.random.choice([0,5], size=(input_nnodes, nnodes), p=[1-p_link, p_link]) # Try other weight values?

      # Create internal weight matrix
      self.link_mat = np.random.choice([0,1], size=(nnodes, nnodes), p=[1-p_link, p_link])
      self.wmat = np.where(self.link_mat == 1, np.random.normal(0, 1, size=(nnodes, nnodes)), 0)

      # Initialize reservoir state
      self.spikes = np.zeros(nnodes)
      self.targets = np.repeat(targ_min, nnodes)
      self.acts = np.zeros(nnodes)
      self.prespike_acts = np.zeros(nnodes)
  
  def get_acts(self, input):
    # Update activation in reservoir
    self.acts = self.acts*self.leak + np.dot(input, self.input_wmat) + np.dot(self.spikes, self.wmat)

    self.prespike_acts = self.acts

    # Log spikes
    thresholds = self.targets*2
    self.spikes[self.acts>=thresholds] = 1
    self.spikes[self.acts<thresholds] = 0

    # Dissipate activation on spiking nodes
    self.acts[self.spikes==1] = self.acts[self.spikes==1] - thresholds[self.spikes==1]
    self.acts[self.acts<0] = 0

    # Log errors
    errors = self.acts-self.targets

    return errors

  def learning(self, prev_spikes, errors):
    # Track which neighbors spiked
    prev_active = np.argwhere(prev_spikes>0)[:,0]
    prev_inactive = np.argwhere(prev_spikes<=0)[:,0]

    # Apply learning rules
    active_neighbors = self.link_mat.copy()
    active_neighbors[prev_inactive,:] = 0
    active_neighbors = np.sum(active_neighbors, axis=0)

    d_wmat = np.zeros((self.nnodes, self.nnodes))
    d_wmat[:,:] = errors*self.lrate_wmat
    d_wmat[self.link_mat==0] = 0
    d_wmat[prev_inactive,:] = 0
    # d_wmat /= active_neighbors
    # d_wmat = np.nan_to_num(d_wmat)

    # print(np.isnan(d_wmat).any())          # Check if there are any NaN values in d_wmat
    # print(np.isnan(active_neighbors).any()) # Check if there are any NaN values in active_neighbors
    # print(np.isinf(d_wmat).any())           # Check for infinite values in d_wmat
    # print(np.isinf(active_neighbors).any())

    d_wmat = np.where(active_neighbors != 0, d_wmat / active_neighbors.astype(np.float64), 0)
    self.wmat -= d_wmat

    self.targets = self.targets + (errors*self.lrate_targ)
    self.targets[self.targets<self.targ_min] = self.targ_min

  def run(self, train_data, learn_on=True):
    # Store data
    log_spikes = pd.DataFrame()
    log_acts = pd.DataFrame()
    log_wmat = pd.DataFrame()

    # Run model on data
    for row in range(len(train_data)):
      prev_spikes = self.spikes.copy()
      input = train_data[row]

      errors = self.get_acts(input)

      if learn_on==True:
        self.learning(prev_spikes, errors)

      log_spikes = pd.concat([log_spikes, pd.Series(self.spikes)], ignore_index=True, axis=1)
      log_acts = pd.concat([log_acts, pd.Series(self.prespike_acts)], ignore_index=True, axis=1) # self.prespike_acts
      log_wmat = pd.concat([log_wmat, pd.Series(self.wmat.flatten())], ignore_index=True, axis=1)

    return log_spikes, log_acts, log_wmat

  def echo(self, cue):
    # Log current state of reservoir
    end_spikes = self.spikes
    end_targets = self.targets
    end_acts = self.acts

    # Store data
    log_spikes = pd.DataFrame()
    log_acts = pd.DataFrame()
    log_wmat = pd.DataFrame()

    # Run model on data
    for row in range(len(cue)):
      prev_spikes = self.spikes.copy()
      input = cue[row]

      errors = self.get_acts(input)
      self.learning(prev_spikes, errors)

      log_spikes = pd.concat([log_spikes, pd.Series(self.spikes)], ignore_index=True, axis=1)
      log_acts = pd.concat([log_acts, pd.Series(self.prespike_acts)], ignore_index=True, axis=1) # self.prespike_acts
      log_wmat = pd.concat([log_wmat, pd.Series(self.wmat.flatten())], ignore_index=True, axis=1)

    # Return state of reservoir
    self.spikes = end_spikes
    self.targets = end_targets
    self.acts = end_acts

    return log_spikes, log_acts, log_wmat
  
  def save(self, filename="net.npz"):
    np.savez(filename, weights=self.wmat)
    print(f"Network saved to {filename}")

  def load(self, filename="net.npz"):
    data = np.load(filename)
    self.wmat = list(data['weights'])
    print(f"Network loaded from {filename}")
