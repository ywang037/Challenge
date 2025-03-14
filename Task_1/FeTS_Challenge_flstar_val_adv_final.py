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
import random
import torch
from fets_challenge import run_challenge_experiment
from fets_challenge.experiment import logger
from pyparsing import col
from sklearn.metrics import log_loss
from pathlib import Path

# the following are customized add-on lib
import time
import argparse

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

##########################################################
# # Custom selection functions 
##########################################################
def select_top6_cols_p2(collaborators,
                    db_iterator,
                    fl_round,
                    collaborators_chosen_each_round,
                    collaborator_times_per_round):
    """ this function selects 6 cols from the hand-picked list of cols who have more than 10 data samples
        the selection is fixed throughout the FL rounds
    """
    
    # this is a list of ids of the 6 collaborators that have more than 100 data samples, in partition_2
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

def lr_schedular_1(collaborators,
                db_iterator,
                fl_round,
                collaborators_chosen_each_round,
                collaborator_times_per_round):
    """Set the learning rate for each fl round.
    
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
    # we leave these parameters unchanged
    epochs_per_round = 1.0
    batches_per_round = None

    init_learning_rate = 5e-5
    
    if fl_round<int(10):
        learning_rate = init_learning_rate # for the first 10 rounds, use default value 
    else:
        learning_rate = 0.5 * init_learning_rate
    return (learning_rate, epochs_per_round, batches_per_round)

def lr_schedular_2(collaborators,
                db_iterator,
                fl_round,
                collaborators_chosen_each_round,
                collaborator_times_per_round):
    """Set the learning rate for each fl round.
    
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
    # we leave these parameters unchanged
    epochs_per_round = 1.0
    batches_per_round = None
    
    init_learning_rate = 5e-5 # intial lr is the same as the default value
    lr_multiplier = [5, 2, 1]
    # lr_multiplier = [10, 1, 0.1]
    # lr_multiplier = [5, 1, 0.2]
    # lr_multiplier = [2, 1, 0.5]
    

    if fl_round<int(5):
        learning_rate = init_learning_rate * lr_multiplier[0]
    elif fl_round<int(9):
        learning_rate = init_learning_rate * lr_multiplier[1] 
    else:
        learning_rate = init_learning_rate * lr_multiplier[1] 
    return (learning_rate, epochs_per_round, batches_per_round)

def lr_schedular_3(collaborators,
                db_iterator,
                fl_round,
                collaborators_chosen_each_round,
                collaborator_times_per_round):
    """Set the learning rate for each fl round.
    
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
    # we leave these parameters unchanged
    epochs_per_round = 1.0
    batches_per_round = None
    
    init_learning_rate = 5e-5 # intial lr is the same as the default value
    lr_decay = 0.9

    if fl_round == 0:
        learning_rate = init_learning_rate
    else:
        learning_rate = init_learning_rate * lr_decay

    return (learning_rate, epochs_per_round, batches_per_round)

##########################################################
# # Custom Aggregation Functions - WY's trials
##########################################################
def get_val_loss_score(local_tensors,tensor_db,fl_round):
    """ this function get the local validation loss of each col
    """
    # metric_name = 'valid_dice'
    metric_name = 'valid_loss'
    tags_local = ('metric','validate_local')
    val_loss = {}
    for _, record in tensor_db.iterrows():
        for t in local_tensors:
            col = t.col_name
            tags = set(tags_local + tuple([col]))
            record_tags = record['tags']

            if (
                tags <= set(record_tags) 
                and record['round'] == fl_round
                and record['tensor_name'] == metric_name
            ):
                val_loss[col]=record['nparray']
    
    # normalize the score to [0, 1]
    sum=0
    for _, loss in val_loss.items():
        sum += loss
    for col, loss in val_loss.items():
        val_loss[col] = loss/sum
    val_scores = np.array([val_loss[t.col_name] for t in local_tensors],dtype=np.float64)
    return val_loss, val_scores

def wy_agg_func_val_adv(local_tensors,
                    tensor_db,
                    tensor_name,
                    fl_round,
                    collaborators_chosen_each_round,
                    collaborator_times_per_round):
    """ advanced aggregation, the weight values are the defualt weights multiply those of customized aggregation function
        this function is similar to eqn (6) in https://arxiv.org/pdf/2203.13993.pdf
    """
    if fl_round == 0:
        # in the first round, just do normal fedavg
        tensor_values = [t.tensor for t in local_tensors]
        weight_values = [t.weight for t in local_tensors]               
        new_tensor_agg = np.average(tensor_values, weights=weight_values, axis=0)   
        
        # besides, also save the aggregated model for the computing of model update direction in the later roudns [optional]
        tensor_db.store(tensor_name=tensor_name, tags=('tensor_agg_round_0',), nparray=new_tensor_agg)
    else:
        # get the defualt weights values (shall be like weighted by number of training samples)
        weight_values_default = np.array([t.weight for t in local_tensors], dtype=np.float64)               

        # compute the customized aggregation weights, i.e., scores associated to each local tensor
        # these scores can be the output of get_dist_score, or get_val_loss, or get_hybrid_score
        _, val_loss_scores = get_val_loss_score(local_tensors,tensor_db,fl_round)

        # compute the final scores
        final_scores = weight_values_default * val_loss_scores

        # normalize the score into [0 ,1]
        weight_values = final_scores/final_scores.sum()
        
        # compute the aggregated model
        tensor_values = [t.tensor for t in local_tensors]
        new_tensor_agg = np.average(tensor_values, weights=weight_values, axis=0)

    return new_tensor_agg

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

def argparser():
    parser = argparse.ArgumentParser(description='FeTS Challenge experiment run')
    parser.add_argument('--seed', type=int, default=0, help='seed for controliing randomness')
    parser.add_argument('--rounds', type=int, default=10, help='number of FL rounds to do')
    parser.add_argument('--restore', type=str, default=None, help='specify the restore folder')
    parser.add_argument('--lr_schedule',type=str, default='const', help='specify the lr schedule')
    return parser.parse_args()

if __name__ == '__main__':  
    
    args = argparser()
    
    # freeze the randomness for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # some baseline aggregation functions
    # aggregation_function = weighted_average_aggregation
    # aggregation_function = FedAvgM_Selection

    # customized aggregation function
    aggregation_function = wy_agg_func_val_adv 

    # training col selection strategy
    # choose_training_collaborators = all_collaborators_train
    choose_training_collaborators = select_top6_cols_p2

    # hyper param.
    if args.lr_schedule == 'const':
        training_hyper_parameters_for_round = constant_hyper_parameters
    elif args.lr_schedule == 'schedular_1':
        training_hyper_parameters_for_round = lr_schedular_1
    elif args.lr_schedule == 'schedular_2':
        training_hyper_parameters_for_round = lr_schedular_2
    elif args.lr_schedule == 'schedular_3':
        training_hyper_parameters_for_round = lr_schedular_3 

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

    # change this to point to the parent directory of the data
    brats_training_data_parent_dir = '/home/wang_yuan/fets2022/Data/TrainingData'

    # increase this if you need a longer history for your algorithms
    # decrease this if you need to reduce system RAM consumption
    db_store_rounds = 1

    # this is passed to PyTorch, so set it accordingly for your system
    device = 'cuda'

    # you'll want to increase this most likely. You can set it as high as you like, 
    # however, the experiment will exit once the simulated time exceeds one week. 
    # rounds_to_train = 10
    rounds_to_train = args.rounds

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

    # infer participant home folder
    home = str(Path.home())
    scores_dataframe_file = os.path.join(home, '.local/workspace/checkpoint', checkpoint_folder, 'scores_df.csv')
    scores_dataframe.to_csv(scores_dataframe_file)

    time_end = time.time()

    # show the time elapsed for this session
    sesseion_time = np.around((time_end-time_start)/3600, 2)
    print('Session time: {} hrs. That\'s all folks.'.format(sesseion_time))

    time_end_stamp = time.strftime('%y-%m-%d-%H-%M-%S')
    print(f'Session completed at {time_end_stamp}')