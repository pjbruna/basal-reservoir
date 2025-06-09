import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from reservoir_linnea import SlimeMoldReservoir  # or your actual module name
from functions import plot_pca_explainedvar, plot_pca_space, plot_correlations, correlate_states

# Hyperparams
simulations = 100
timesteps = 250

# Create oscillatory regime
train_data = [[1], [0]] * (timesteps // 2)
train_labels = ['ON', 'OFF'] * (timesteps // 2)
 
for run in range(simulations):

    # Create and train the model
    model = SlimeMoldReservoir(
        input_nnodes=len(train_data[0]),
        nnodes=36,
        input_connectivity=1.0,
        leak=0.01,
        p_link=0.2,
        lrate_targ=0.01,
        lrate_wmat=0.01,
        targ_min=1.0,
        network_type='hexagonal',  # you can also try 'grid', 'hexagonal', or 'random'
        seed=None,
        decay_rate=0.00
    )


    # Run training
    results = model.run(train_data)


    # Cue trained model
    cue = [[0], [0]] * 5
    cue_labels = ['NA', 'NA'] * 5

    cue_results = model.echo(cue)


    # Save data

    actdf_train = pd.DataFrame(results[0].T)
    actdf_cue = pd.DataFrame(cue_results[0].T)

    wdf_train = pd.DataFrame(results[1].T)
    wdf_cue = pd.DataFrame(cue_results[1].T)
    
    actdf_train.to_csv(f'data_mem250/act_train_sim_{run}.csv', index=False)
    actdf_cue.to_csv(f'data_mem250/act_cue_sim_{run}.csv', index=False)

    wdf_train.to_csv(f'data_mem250/w_train_sim_{run}.csv', index=False)
    wdf_cue.to_csv(f'data_mem250/w_cue_sim_{run}.csv', index=False)