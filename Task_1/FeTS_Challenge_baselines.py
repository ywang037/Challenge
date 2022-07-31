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
    learning_rate = 5e-5
    return (learning_rate, epochs_per_round, batches_per_round)

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
    parser.add_argument('--method', type=str, default='fedavg', help='speicify the aggregation function')
    parser.add_argument('--seed', type=int, default=0, help='seed for controliing randomness')
    parser.add_argument('--rounds', type=int, default=10, help='number of FL rounds to do')
    parser.add_argument('--restore', type=str, default=None, help='specify the restore folder')
    parser.add_argument('--inference', default=False, action='store_true', help='if to do inference after training')
    return parser.parse_args()

if __name__ == '__main__':  
    
    args = argparser()
    
    # freeze the randomness
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # specify aggregation function
    if args.method == 'fedavg':
        aggregation_function = weighted_average_aggregation
    elif args.method == 'fedavgm':
        aggregation_function = FedAvgM_Selection

    training_hyper_parameters_for_round = constant_hyper_parameters
    choose_training_collaborators = all_collaborators_train
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
    # rounds_to_train = 15
    rounds_to_train = args.rounds

    # (bool) Determines whether checkpoints should be saved during the experiment. 
    # The checkpoints can grow quite large (5-10GB) so only the latest will be saved when this parameter is enabled
    save_checkpoints = True

    # path to previous checkpoint folder for experiment that was stopped before completion. 
    # Checkpoints are stored in ~/.local/workspace/checkpoint, and you should provide the experiment directory 
    # relative to this path (i.e. 'experiment_1'). Please note that if you restore from a checkpoint, 
    # and save checkpoint is set to True, then the checkpoint you restore from will be subsequently overwritten.
    # restore_from_checkpoint_folder = 'experiment_1'
    # restore_from_checkpoint_folder = None
    restore_from_checkpoint_folder = args.restore


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

    home = str(Path.home())
    scores_dataframe_file = os.path.join(home, '.local/workspace/checkpoint', checkpoint_folder, 'scores_df.csv')
    scores_dataframe.to_csv(scores_dataframe_file)
    
    time_end = time.time()
    # show the time elapsed for this session
    sesseion_time = np.around((time_end-time_start)/3600, 2)
    print('Session time: {} hrs. That\'s all folks.'.format(sesseion_time))

    time_end_stamp = time.strftime('%y-%m-%d-%H-%M-%S')
    print(f'Session completed at {time_end_stamp}')

    # infer participant home folder
    if args.inference:
        from fets_challenge import model_outputs_to_disc
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

