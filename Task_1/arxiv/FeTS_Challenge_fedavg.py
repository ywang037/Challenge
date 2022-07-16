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


from math import floor
import os
import numpy as np

from fets_challenge import run_challenge_experiment

# the following are customized add-on lib
import time

# # Adding custom functionality to the experiment
# Within this notebook there are **four** functional areas that you can adjust to improve upon the challenge reference code:
# 
# - [Custom aggregation logic](#Custom-Aggregation-Functions)
# - [Selection of training hyperparameters by round](#Custom-hyperparameters-for-training)
# - [Collaborator training selection by round](#Custom-Collaborator-Training-Selection)
# 

# ## Experiment logger for your functions
# The following import allows you to use the same logger used by the experiment framework. This lets you include logging in your functions.

from fets_challenge.experiment import logger
from pyparsing import col
from sklearn.metrics import log_loss

# # Custom Collaborator Training Selection
# By default, all collaborators will be selected for training each round, 
# but you can easily add your own logic to select a different set of collaborators based on custom criteria. 
# An example is provided below for selecting a single collaborator on odd rounds that had the fastest training time (`one_collaborator_on_odd_rounds`).


# a very simple function. Everyone trains every round.
def all_collaborators_train(collaborators,
                            db_iterator,
                            fl_round,
                            collaborators_chosen_each_round,
                            collaborator_times_per_round):
    """Chooses which collaborators will train for a given round.
    
    Args:
        collaborators: list of strings of collaborator names
        db_iterator: iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray']
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.  
    """
    return collaborators

# this is not a good algorithm, but we include it to demonstrate the following:
    # simple use of the logger and of fl_round
    # you can search through the "collaborator_times_per_round" dictionary to see how long collaborators have been taking
    # you can have a subset of collaborators train in a given round
def one_collaborator_on_odd_rounds(collaborators,
                                   db_iterator,
                                   fl_round,
                                   collaborators_chosen_each_round,
                                   collaborator_times_per_round):
    """Chooses which collaborators will train for a given round.
    
    Args:
        collaborators: list of strings of collaborator names
        db_iterator: iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray']
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.  
    """
    logger.info("one_collaborator_on_odd_rounds called!")
    # on odd rounds, choose the fastest from the previous round
    if fl_round % 2 == 1:
        training_collaborators = None
        fastest_time = np.inf
        
        # the previous round information will be index [fl_round - 1]
        # this information is itself a dictionary of {col: time}
        for col, t in collaborator_times_per_round[fl_round - 1].items():
            if t < fastest_time:
                fastest_time = t
                training_collaborators = [col]
    else:
        training_collaborators = collaborators
    return training_collaborators

##########################################################
# # Custom selection functions - WY's trials
##########################################################

def wy_select_col_with_more_data_1(collaborators,
                                    db_iterator,
                                    fl_round,
                                    collaborators_chosen_each_round,
                                    collaborator_times_per_round):
    # this is a list of ids for collaborator that has at least 10 data samples, hand-picked
    training_collaborators = ['1', '3', '4', '5', '6', '7', '11', '12', '13', '15', '16', '18', '20', '21']
    
    return training_collaborators

def wy_select_col_with_more_data_2(collaborators,
                                    db_iterator,
                                    fl_round,
                                    collaborators_chosen_each_round,
                                    collaborator_times_per_round):
    # this is a list of ids for collaborator that has at least 10 data samples, hand-picked
    training_collaborators = preserved_col_id = ['1', '2', '3', '5', '6', '7', '8', '9', '10', '11', '15', '16', '17', '18', '19', '21', '22', '24', '25', '26', '28', '29', '30', '31'] # for partition_2
    return training_collaborators
   
def random_sel_more_data_subset_p2(collaborators,
                                    db_iterator,
                                    fl_round,
                                    collaborators_chosen_each_round,
                                    collaborator_times_per_round):
    """ this function randomly select training cols. from the hand-picked subset of all cols. with more than 10 data samples
        the random selection using the probabilities which are normalized number of training samples 
    """
    
    # this is a list of ids for collaborator that has at least 10 data samples, hand-picked
    preserved_col_id = ['1', '2', '3', '5', '6', '7', '8', '9', '10', '11', '15', '16', '17', '18', '19', '21', '22', '24', '25', '26', '28', '29', '30', '31']
    
    # hand-calculated problabilities, for partition_2
    prob_sel = [
        0.14237856, 0.14237856, 0.14321608, 0.01256281, 0.01340034, 0.01256281,
        0.01340034, 0.01842546, 0.02847571, 0.01005025, 0.01172529, 0.00921273,
        0.01005025, 0.00921273, 0.01005025, 0.01088777, 0.02512563, 0.10636516,
        0.10636516, 0.10720268, 0.02763819, 0.01005025, 0.00921273, 0.01005025
        ]
    prob_sel = np.array(prob_sel, dtype=np.float32) # make it a np array

    rng = np.random.default_rng()
    num_final_pick = 8 # 8 out of 33 approximately 25% ratio, 6 out of 33 approx. 20% ratio
    training_collaborators = rng.choice(preserved_col_id, num_final_pick, replace=False, p=prob_sel)
    
    return training_collaborators

def random_sel_more_data_p2(collaborators,
                            db_iterator,
                            fl_round,
                            collaborators_chosen_each_round,
                            collaborator_times_per_round):
    """ this function randomly selects training cols. from partition_2, 
        using the probabilities calculated as the normalized number of data samples
        so cols. have more data are more likely to be selected.
    """
    col_ids = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
        '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
        '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
        '31', '32', '33'
        ]
    prob_sel = [
        0.13589129, 0.13589129, 0.13669065, 0.00479616, 0.01199041, 0.01278977,
        0.01199041, 0.01278977, 0.01758593, 0.02717826, 0.00959233, 0.00639488,
        0.00319744, 0.00639488, 0.01119105, 0.00879297, 0.00959233, 0.00879297,
        0.00959233, 0.00479616, 0.01039169, 0.02398082, 0.00719424, 0.10151878,
        0.10151878, 0.10231815, 0.00319744, 0.0263789,  0.00959233, 0.00879297,
        0.00959233, 0.00559552, 0.0039968
        ]
    prob_sel = np.array(prob_sel, dtype=np.float32) # make it a np array

    rng = np.random.default_rng()
    num_final_pick = 6 # 8 out of 33 approximately 25% ratio, 6 out of 33 approx. 20% ratio
    training_collaborators = rng.choice(col_ids, num_final_pick, replace=False, p=prob_sel)
    
    return training_collaborators

def select_top6_cols_p2(collaborators,
                    db_iterator,
                    fl_round,
                    collaborators_chosen_each_round,
                    collaborator_times_per_round):
    """ this function randomly select a subset of collaborators from the hand-picked list of collaborators with more than 10 data samples
        the random selection using the probabilities which are normalized number of training samples 
    """
    
    # this is a list of ids of the 5 collaborators that have more than 100 data samples, in partition_2
    training_collaborators = ['1', '2', '3', '24', '25', '26']
    
    return training_collaborators
##############################################
# Custom hyperparameters for training
###############################################

# This simple example uses constant hyper-parameters through the experiment
def constant_hyper_parameters(collaborators,
                              db_iterator,
                              fl_round,
                              collaborators_chosen_each_round,
                              collaborator_times_per_round):
    """Set the training hyper-parameters for the round.
    
    Args:
        collaborators: list of strings of collaborator names
        db_iterator: iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray']
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.  
    Returns:
        tuple of (learning_rate, epochs_per_round, batches_per_round). One of epochs_per_round and batches_per_round must be None.
    """
    # these are the hyperparameters used in the May 2021 recent training of the actual FeTS Initiative
    # they were tuned using a set of data that UPenn had access to, not on the federation itself
    # they worked pretty well for us, but we think you can do better :)
    epochs_per_round = 1.0
    batches_per_round = None
    # batches_per_round = 1
    learning_rate = 5e-5
    return (learning_rate, epochs_per_round, batches_per_round)

def lr_schedular(collaborators,
                db_iterator,
                fl_round,
                collaborators_chosen_each_round,
                collaborator_times_per_round):
    """Set the training hyper-parameters for the round.
    
    Args:
        collaborators: list of strings of collaborator names
        db_iterator: iterator over history of all tensors.
            Columns: ['tensor_name', 'round', 'tags', 'nparray']
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.  
    Returns:
        tuple of (learning_rate, epochs_per_round, batches_per_round). One of epochs_per_round and batches_per_round must be None.
    """
    epochs_per_round = 1.0
    batches_per_round = None
    init_learning_rate = 5e-5
    if fl_round<int(10):
        learning_rate = init_learning_rate # for the first 10 rounds, use default value 
    else:
        learning_rate = 0.5 * init_learning_rate
    return (learning_rate, epochs_per_round, batches_per_round)


#############################################################
##         BASELINE METHODS
##############################################################

# the simple example of weighted FedAVG
def weighted_average_aggregation(local_tensors,
                                 tensor_db,
                                 tensor_name,
                                 fl_round,
                                 collaborators_chosen_each_round,
                                 collaborator_times_per_round):
    """Aggregate tensors. This aggregator clips all tensor values to the 80th percentile of the absolute values to prevent extreme changes.

    Args:
        local_tensors(list[openfl.utilities.LocalTensor]): List of local tensors to aggregate.
        tensor_db: pd.DataFrame that contains global tensors / metrics.
            Columns: ['tensor_name', 'origin', 'round', 'report',  'tags', 'nparray']
        tensor_name: name of the tensor
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.
    """
    # basic weighted fedavg

    # here are the tensor values themselves
    tensor_values = [t.tensor for t in local_tensors]
    
    # and the weights (i.e. data sizes)
    weight_values = [t.weight for t in local_tensors]
    
    # so we can just use numpy.average
    return np.average(tensor_values, weights=weight_values, axis=0)

# here we will clip outliers by clipping deltas to the Nth percentile (e.g. 80th percentile)
def clipped_aggregation(local_tensors,
                        tensor_db,
                        tensor_name,
                        fl_round,
                        collaborators_chosen_each_round,
                        collaborator_times_per_round):
    """Aggregate tensors. This aggregator clips all tensor values to the 80th percentile of the absolute values to prevent extreme changes.

    Args:
        local_tensors(list[openfl.utilities.LocalTensor]): List of local tensors to aggregate.
        tensor_db: pd.DataFrame that contains global tensors / metrics.
            Columns: ['tensor_name', 'origin', 'round', 'report',  'tags', 'nparray']
        tensor_name: name of the tensor
        fl_round: round number
        collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
        collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.
    """
    # the percentile we will clip to
    clip_to_percentile = 80
    
    # first, we need to determine how much each local update has changed the tensor from the previous value
    # we'll use the tensor_db search function to find the 
    previous_tensor_value = tensor_db.search(tensor_name=tensor_name, fl_round=fl_round, tags=('model',), origin='aggregator')

    if previous_tensor_value.shape[0] > 1:
        print(previous_tensor_value)
        raise ValueError(f'found multiple matching tensors for {tensor_name}, tags=(model,), origin=aggregator')

    if previous_tensor_value.shape[0] < 1:
        # no previous tensor, so just return the weighted average
        return weighted_average_aggregation(local_tensors,
                                            tensor_db,
                                            tensor_name,
                                            fl_round,
                                            collaborators_chosen_each_round,
                                            collaborator_times_per_round)

    previous_tensor_value = previous_tensor_value.nparray.iloc[0]

    # compute the deltas for each collaborator 
    # NOTE you may re-use this line for your purpose
    deltas = [t.tensor - previous_tensor_value for t in local_tensors]

    # get the target percentile using the absolute values of the deltas
    clip_value = np.percentile(np.abs(deltas), clip_to_percentile)
        
    # let's log what we're clipping to
    logger.info("Clipping tensor {} to value {}".format(tensor_name, clip_value))
    
    # now we can compute our clipped tensors
    clipped_tensors = []
    for delta, t in zip(deltas, local_tensors):
        new_tensor = previous_tensor_value + np.clip(delta, -1 * clip_value, clip_value)
        clipped_tensors.append(new_tensor)
        
    # get an array of weight values for the weighted average
    weights = [t.weight for t in local_tensors]

    # return the weighted average of the clipped tensors
    return np.average(clipped_tensors, weights=weights, axis=0)

# Adapted from FeTS Challenge 2021
# Federated Brain Tumor Segmentation:Multi-Institutional Privacy-Preserving Collaborative Learning
# Ece Isik-Polat, Gorkem Polat,Altan Kocyigit1, and Alptekin Temizel1
def FedAvgM_Selection(local_tensors,
                      tensor_db,
                      tensor_name,
                      fl_round,
                      collaborators_chosen_each_round,
                      collaborator_times_per_round):
    
        """Aggregate tensors.

        Args:
            local_tensors(list[openfl.utilities.LocalTensor]): List of local tensors to aggregate.
            tensor_db: Aggregator's TensorDB [writable]. Columns:
                - 'tensor_name': name of the tensor.
                    Examples for `torch.nn.Module`s: 'conv1.weight', 'fc2.bias'.
                - 'round': 0-based number of round corresponding to this tensor.
                - 'tags': tuple of tensor tags. Tags that can appear:
                    - 'model' indicates that the tensor is a model parameter.
                    - 'trained' indicates that tensor is a part of a training result.
                        These tensors are passed to the aggregator node after local learning.
                    - 'aggregated' indicates that tensor is a result of aggregation.
                        These tensors are sent to collaborators for the next round.
                    - 'delta' indicates that value is a difference between rounds
                        for a specific tensor.
                    also one of the tags is a collaborator name
                    if it corresponds to a result of a local task.

                - 'nparray': value of the tensor.
            tensor_name: name of the tensor
            fl_round: round number
        Returns:
            np.ndarray: aggregated tensor
        """
        #momentum
        tensor_db.store(tensor_name='momentum',nparray=0.9,overwrite=False)
        #aggregator_lr
        tensor_db.store(tensor_name='aggregator_lr',nparray=1.0,overwrite=False)

        if fl_round == 0:
            # Just apply FedAvg

            tensor_values = [t.tensor for t in local_tensors]
            weight_values = [t.weight for t in local_tensors]               
            new_tensor_weight =  np.average(tensor_values, weights=weight_values, axis=0)        

            #if not (tensor_name in weight_speeds):
            if tensor_name not in tensor_db.search(tags=('weight_speeds',))['tensor_name']:    
                #weight_speeds[tensor_name] = np.zeros_like(local_tensors[0].tensor) # weight_speeds[tensor_name] = np.zeros(local_tensors[0].tensor.shape)
                tensor_db.store(
                    tensor_name=tensor_name, 
                    tags=('weight_speeds',), 
                    nparray=np.zeros_like(local_tensors[0].tensor),
                )
            return new_tensor_weight        
        else:
            if tensor_name.endswith("weight") or tensor_name.endswith("bias"): # NOTE why would like to have this?
                # Calculate aggregator's last value
                previous_tensor_value = None
                for _, record in tensor_db.iterrows():
                    if (record['round'] == fl_round 
                        and record["tensor_name"] == tensor_name
                        and record["tags"] == ("aggregated",)): 
                        previous_tensor_value = record['nparray']
                        break

                if previous_tensor_value is None:
                    logger.warning("Error in fedAvgM: previous_tensor_value is None")
                    logger.warning("Tensor: " + tensor_name)

                    # Just apply FedAvg       
                    tensor_values = [t.tensor for t in local_tensors]
                    weight_values = [t.weight for t in local_tensors]               
                    new_tensor_weight =  np.average(tensor_values, weights=weight_values, axis=0)        
                    
                    if tensor_name not in tensor_db.search(tags=('weight_speeds',))['tensor_name']:    
                        tensor_db.store(
                            tensor_name=tensor_name, 
                            tags=('weight_speeds',), 
                            nparray=np.zeros_like(local_tensors[0].tensor),
                        )

                    return new_tensor_weight
                else:
                    # compute the average delta for that layer
                    deltas = [previous_tensor_value - t.tensor for t in local_tensors] # NOTE you may wish to re-use this line
                    weight_values = [t.weight for t in local_tensors]
                    average_deltas = np.average(deltas, weights=weight_values, axis=0) 

                    # V_(t+1) = momentum*V_t + Average_Delta_t
                    tensor_weight_speed = tensor_db.retrieve(
                        tensor_name=tensor_name,
                        tags=('weight_speeds',)
                    )
                    
                    momentum = float(tensor_db.retrieve(tensor_name='momentum'))
                    aggregator_lr = float(tensor_db.retrieve(tensor_name='aggregator_lr'))
                    
                    new_tensor_weight_speed = momentum * tensor_weight_speed + average_deltas # fix delete (1-momentum)
                    
                    tensor_db.store(
                        tensor_name=tensor_name, 
                        tags=('weight_speeds',), 
                        nparray=new_tensor_weight_speed
                    )
                    # W_(t+1) = W_t-lr*V_(t+1)
                    new_tensor_weight = previous_tensor_value - aggregator_lr*new_tensor_weight_speed

                    return new_tensor_weight
            else:
                # Just apply FedAvg       
                tensor_values = [t.tensor for t in local_tensors]
                weight_values = [t.weight for t in local_tensors]               
                new_tensor_weight =  np.average(tensor_values, weights=weight_values, axis=0)

                return new_tensor_weight


def fedNova_simplified(local_tensors,
                        tensor_db,
                        tensor_name,
                        fl_round,
                        collaborators_chosen_each_round,
                        collaborator_times_per_round):
    
    aggregator_lr = 1.0

    if fl_round == 0:
        # Just apply FedAvg
        
        tensor_values = [t.tensor for t in local_tensors]
        weight_values = [t.weight for t in local_tensors]               
        new_tensor_weight =  np.average(tensor_values, weights=weight_values, axis=0)        
                       
        return new_tensor_weight  
    else:
        # Calculate aggregator's last value
        previous_tensor_value = None
        for _, record in tensor_db.iterrows():
            if (record['round'] == (fl_round) 
                and record["tensor_name"] == tensor_name
                and record["tags"] == ("aggregated",)):
                previous_tensor_value = record['nparray']
                break
                
        deltas = [previous_tensor_value - t.tensor for t in local_tensors]
#         weight_values = [t.weight for t in local_tensors]
        grad_nova =  np.average(deltas, axis=0)
        
        new_tensor_weight = previous_tensor_value - aggregator_lr *grad_nova
        
        return new_tensor_weight

# # Running the Experiment
# 
# ```run_challenge_experiment``` is singular interface where your custom methods can be passed.
# 
# - ```aggregation_function```, ```choose_training_collaborators```, and ```training_hyper_parameters_for_round``` correspond to the [this list](#Custom-hyperparameters-for-training) of configurable functions 
# described within this notebook.
# - ```institution_split_csv_filename``` : Describes how the data should be split between all collaborators. Extended documentation about configuring the splits in the ```institution_split_csv_filename``` parameter can be found in the [README.md](https://github.com/FETS-AI/Challenge/blob/main/Task_1/README.md). 
# - ```db_store_rounds``` : This parameter determines how long metrics and weights should be stored by the aggregator before being deleted. Providing a value of `-1` will result in all historical data being retained, but memory usage will likely increase.
# - ```rounds_to_train``` : Defines how many rounds will occur in the experiment
# - ```device``` : Which device to use for training and validation

# ## Setting up the experiment
# Now that we've defined our custom functions, the last thing to do is to configure the experiment. The following cell shows the various settings you can change in your experiment.
# 
# Note that ```rounds_to_train``` can be set as high as you want. However, the experiment will exit once the simulated time value exceeds 1 week of simulated time, or if the specified number of rounds has completed.


# change any of these you wish to your custom functions. You may leave defaults if you wish.
aggregation_function = weighted_average_aggregation
# aggregation_function = FedAvgM_Selection
# aggregation_function = fedNova_simplified

# training col selection strategy
choose_training_collaborators = all_collaborators_train
# choose_training_collaborators = select_top6_cols_p2

# hyper param.
training_hyper_parameters_for_round = constant_hyper_parameters
# training_hyper_parameters_for_round = lr_schedular

# As mentioned in the 'Custom Aggregation Functions' section (above), six 
# perfomance evaluation metrics are included by default for validation outputs in addition 
# to those you specify immediately above. Changing the below value to False will change 
# this fact, excluding the three hausdorff measurements. As hausdorff distance is 
# expensive to compute, excluding them will speed up your experiments.
include_validation_with_hausdorff=False

# We encourage participants to experiment with partitioning_1 and partitioning_2, as well as to create
# other partitionings to test your changes for generalization to multiple partitionings.
# institution_split_csv_filename = 'small_split.csv'
# institution_split_csv_filename = 'partitioning_1.csv'
institution_split_csv_filename = 'partitioning_2.csv'
# institution_split_csv_filename = 'partitioning_2_top5_clients.csv'
# institution_split_csv_filename = 'partitioning_2_rand_pick_5.csv'


# change this to point to the parent directory of the data
brats_training_data_parent_dir = '/home/wang_yuan/fets2022/Data/TrainingData'

# increase this if you need a longer history for your algorithms
# decrease this if you need to reduce system RAM consumption
db_store_rounds = 1

# this is passed to PyTorch, so set it accordingly for your system
device = 'cuda'

# you'll want to increase this most likely. You can set it as high as you like, 
# however, the experiment will exit once the simulated time exceeds one week. 
rounds_to_train = 10

# (bool) Determines whether checkpoints should be saved during the experiment. 
# The checkpoints can grow quite large (5-10GB) so only the latest will be saved when this parameter is enabled
save_checkpoints = True

# path to previous checkpoint folder for experiment that was stopped before completion. 
# Checkpoints are stored in ~/.local/workspace/checkpoint, and you should provide the experiment directory 
# relative to this path (i.e. 'experiment_1'). Please note that if you restore from a checkpoint, 
# and save checkpoint is set to True, then the checkpoint you restore from will be subsequently overwritten.
# restore_from_checkpoint_folder = 'experiment_1'
restore_from_checkpoint_folder = None


time_start = time.time()

# the scores are returned in a Pandas dataframe
scores_dataframe, checkpoint_folder = run_challenge_experiment(
    aggregation_function=aggregation_function,
    choose_training_collaborators=choose_training_collaborators,
    training_hyper_parameters_for_round=training_hyper_parameters_for_round,
    include_validation_with_hausdorff=include_validation_with_hausdorff,
    institution_split_csv_filename=institution_split_csv_filename,
    brats_training_data_parent_dir=brats_training_data_parent_dir,
    db_store_rounds=db_store_rounds,
    rounds_to_train=rounds_to_train,
    device=device,
    save_checkpoints=save_checkpoints,
    restore_from_checkpoint_folder = restore_from_checkpoint_folder)


scores_dataframe
print(scores_dataframe)

from pathlib import Path
# infer participant home folder
home = str(Path.home())
scores_dataframe_file = os.path.join(home, '.local/workspace/checkpoint', checkpoint_folder, 'scores_df.csv')
scores_dataframe.to_csv(scores_dataframe_file)


# ## Produce NIfTI files for best model outputs on the validation set
# Now we will produce model outputs to submit to the leader board.
# 
# At the end of every experiment, the best model (according to average ET, TC, WT DICE) 
# is saved to disk at: ~/.local/workspace/checkpoint/\<checkpoint folder\>/best_model.pkl,
# where \<checkpoint folder\> is the one printed to stdout during the start of the 
# experiment (look for the log entry: "Created experiment folder experiment_##..." above).


# from fets_challenge import model_outputs_to_disc
# from pathlib import Path

# # infer participant home folder
# home = str(Path.home())

# # you will need to specify the correct experiment folder and the parent directory for
# # the data you want to run inference over (assumed to be the experiment that just completed)

# #checkpoint_folder='experiment_1'
# #data_path = </PATH/TO/CHALLENGE_VALIDATION_DATA>
# data_path = '/home/wang_yuan/fets2022/Data/ValidationData'
# validation_csv_filename = 'validation.csv'

# # you can keep these the same if you wish
# final_model_path = os.path.join(home, '.local/workspace/checkpoint', checkpoint_folder, 'best_model.pkl')

# # If the experiment is only run for a single round, use the temp model instead
# if not Path(final_model_path).exists():
#    final_model_path = os.path.join(home, '.local/workspace/checkpoint', checkpoint_folder, 'temp_model.pkl')

# outputs_path = os.path.join(home, '.local/workspace/checkpoint', checkpoint_folder, 'model_outputs')


# # Using this best model, we can now produce NIfTI files for model outputs 
# # using a provided data directory

# model_outputs_to_disc(data_path=data_path, 
#                       validation_csv=validation_csv_filename,
#                       output_path=outputs_path, 
#                       native_model_path=final_model_path,
#                       outputtag='',
#                       device=device)

time_end = time.time()

# show the time elapsed for this session
sesseion_time = np.around((time_end-time_start)/3600, 2)
print('Session time: {} hrs. That\'s all folks.'.format(sesseion_time))

time_end_stamp = time.strftime('%y-%m-%d-%H-%M-%S')
print(f'Session completed at {time_end_stamp}')