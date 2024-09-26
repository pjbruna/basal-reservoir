import numpy as np
import pandas as pd
import random

## MODEL ##

class ResonanceNetwork:
  def __init__(self, input_nnodes, nnodes, p_link, leak, lrate_targ, lrate_wmat, targ_min):
    # Model hyperparameters
    self.nnodes = nnodes
    self.p_link = p_link
    self.leak = leak
    self.lrate_targ = lrate_targ
    self.lrate_wmat = lrate_wmat
    self.targ_min = targ_min

    # Creat input layer and weights
    self.input_wmat = np.random.choice([0,5], size=(input_nnodes, nnodes), p=[1-p_link, p_link]) # Try other weight values?

    # Create internal weight matrix
    self.link_mat = np.random.choice([0,1], size=(nnodes, nnodes), p=[1-p_link, p_link])
    self.wmat = np.where(self.link_mat == 1, np.random.normal(0, 1, size=(nnodes, nnodes)), 0)

    # Initialize reservoir state
    self.spikes = np.zeros(nnodes)
    self.targets = np.repeat(targ_min, nnodes)
    self.acts = np.zeros(nnodes)
  
  def get_acts(self, input):
    # Update activation in reservoir
    self.acts = self.acts*self.leak + np.dot(input, self.input_wmat) + np.dot(self.spikes, self.wmat)

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
    d_wmat /= active_neighbors
    d_wmat = np.nan_to_num(d_wmat)
    self.wmat -= d_wmat

    self.targets = self.targets + (errors*self.lrate_targ)
    self.targets[self.targets<self.targ_min] = self.targ_min

  def run(self, train_data):
    # Store data
    log_spikes = pd.DataFrame()

    # Run model on data
    for row in range(len(train_data)):
      prev_spikes = self.spikes.copy()
      input = train_data[row]

      errors = self.get_acts(input)
      self.learning(prev_spikes, errors)

      log_spikes = pd.concat([log_spikes, pd.Series(self.spikes)], ignore_index=True, axis=1)

    return log_spikes

  def echo(self, cue):
    # Log current state of reservoir
    end_spikes = self.spikes
    end_targets = self.targets
    end_acts = self.acts

    # Store data
    log_spikes = pd.DataFrame()

    # Run model on data
    for row in range(len(cue)):
      prev_spikes = self.spikes.copy()
      input = cue[row]

      errors = self.get_acts(input)
      self.learning(prev_spikes, errors)

      log_spikes = pd.concat([log_spikes, pd.Series(self.spikes)], ignore_index=True, axis=1)

    # Return state of reservoir
    self.spikes = end_spikes
    self.targets = end_targets
    self.acts = end_acts

    return log_spikes