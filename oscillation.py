import numpy as np
import pandas as pd
import random
from reservoir import *
from functions import *

# Create training data

train_data = [[0,1], [1,0]] * 1000
train_labels = ['1', '2'] * 1000

# Train model

model = SlimeMoldReservoir(input_nnodes=len(train_data[0]), nnodes=10, p_link=0.1, leak=0.75, lrate_targ=0.01, lrate_wmat=0.1, targ_min=1)
results = model.run(train_data)

# Cue trained model

state = 0 # select matrix for correlations: 0 - activations, 1 - reservoir weights

cue = [[0,0], [0,0], [0,0], [0,0]]
cue_labels = ['NA', 'NA', 'NA', 'NA']

cue_results = model.echo(cue)

print(cue_results[state].sum(axis=0)) # fading memory

cue1_cor_matrix = correlate_states(results[state], cue_results[state])
plot_correlations(cue1_cor_matrix, train_labels, cue_labels)
