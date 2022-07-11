#!/usr/bin/env python
# coding: utf-8

# # FeTS Challenge
# 
# Contributing Authors (alphabetical order):
# - Brandon Edwards (Intel)
# - Patrick Foley (Intel)
# - Alexey Gruzdev (Intel)
# - Sarthak Pati (University of Pennsylvania)
# - Micah Sheller (Intel)
# - Ilya Trushkin (Intel)



import os


# # Adding custom functionality to the experiment
# Within this notebook there are **four** functional areas that you can adjust to improve upon the challenge reference code:
# 
# - [Custom aggregation logic](#Custom-Aggregation-Functions)
# - [Selection of training hyperparameters by round](#Custom-hyperparameters-for-training)
# - [Collaborator training selection by round](#Custom-Collaborator-Training-Selection)
# 

# ## Experiment logger for your functions
# The following import allows you to use the same logger used by the experiment framework. This lets you include logging in your functions.



# # Custom Collaborator Training Selection
# By default, all collaborators will be selected for training each round, 
# but you can easily add your own logic to select a different set of collaborators based on custom criteria. 
# An example is provided below for selecting a single collaborator on odd rounds that had the fastest training time (`one_collaborator_on_odd_rounds`).


## Produce NIfTI files for best model outputs on the validation set
# Now we will produce model outputs to submit to the leader board.

# At the end of every experiment, the best model (according to average ET, TC, WT DICE) 
# is saved to disk at: ~/.local/workspace/checkpoint/\<checkpoint folder\>/best_model.pkl,
# where \<checkpoint folder\> is the one printed to stdout during the start of the 
# experiment (look for the log entry: "Created experiment folder experiment_##..." above).

# this is passed to PyTorch, so set it accordingly for your system
device = 'cuda'

from fets_challenge import model_outputs_to_disc
from pathlib import Path

# infer participant home folder
home = str(Path.home())

# you will need to specify the correct experiment folder and the parent directory for
# the data you want to run inference over (assumed to be the experiment that just completed)

checkpoint_folder='experiment_1'
#data_path = </PATH/TO/CHALLENGE_VALIDATION_DATA>

data_path = '/home/wang_yuan/fets2022/Data/ValidationData'
validation_csv_filename = 'validation.csv'

# you can keep these the same if you wish
final_model_path = os.path.join(home, '.local/workspace/checkpoint', checkpoint_folder, 'best_model.pkl')

# If the experiment is only run for a single round, use the temp model instead
if not Path(final_model_path).exists():
   final_model_path = os.path.join(home, '.local/workspace/checkpoint', checkpoint_folder, 'temp_model.pkl')

outputs_path = os.path.join(home, '.local/workspace/checkpoint', checkpoint_folder, 'model_outputs')


# Using this best model, we can now produce NIfTI files for model outputs 
# using a provided data directory

model_outputs_to_disc(data_path=data_path, 
                      validation_csv=validation_csv_filename,
                      output_path=outputs_path, 
                      native_model_path=final_model_path,
                      outputtag='',
                      device=device)
