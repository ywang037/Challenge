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

# Participant's Notes (as of 15-Jul-2022)
# This script contains the code submitted to FeTS Challenge 2022 for the evaluation phase
# The code contains functions developed by team FLSAR from Institute of Higher Performance Computing, Singapore

from math import floor
import os
import numpy as np
import random
import torch
from fets_challenge import run_challenge_experiment
from fets_challenge.experiment import logger
from pyparsing import col
from sklearn.metrics import log_loss

from fets_challenge import model_outputs_to_disc
from pathlib import Path

# the following are customized add-on lib
import time

###########################################
# Custom-Collaborator-Training-Selection  #
###########################################
def sel_top6(collaborators,
            db_iterator,
            fl_round,
            collaborators_chosen_each_round,
            collaborator_times_per_round):
    """ this function selects 6 cols that have more than 100 data samples (patient records)
        the selection is fixed throughout the FL rounds
    """
    
    # this is a list of ids of the 6 collaborators that have more than 100 data samples, in partition_2
    training_collaborators = ['1', '2', '3', '24', '25', '26']
    return training_collaborators


#######################################
# Custom-hyperparameters-for-training #
#######################################

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
    # we have used the following default setup for training hyper-parameters
    epochs_per_round = 1.0
    batches_per_round = None
    learning_rate = 5e-5
    return (learning_rate, epochs_per_round, batches_per_round)

################################
# Custom-Aggregation-Functions #
################################
def get_val_loss_score(local_tensors,tensor_db,fl_round):
    """ this function get the local validation loss of each col
    """
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

def val_mul(local_tensors,
                    tensor_db,
                    tensor_name,
                    fl_round,
                    collaborators_chosen_each_round,
                    collaborator_times_per_round):
    """ here is the function implementing the method 'Val-Mul' in the related short paper
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


# below is the main program
if __name__ == '__main__':  
       
    # freeze the randomness for reproducibility
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # customized aggregation function
    aggregation_function = val_mul 

    # training col selection strategy
    choose_training_collaborators = sel_top6

    # training hyper param.
    training_hyper_parameters_for_round = constant_hyper_parameters    

    # other setups
    include_validation_with_hausdorff=False   
    db_store_rounds = 1
    device = 'cuda'
    rounds_to_train = 20 # we hit the simulation time limit after 16-th round
    
    # checkpoints related
    save_checkpoints = True
    restore_from_checkpoint_folder = None

    # pls change do the correct data and file path
    brats_training_data_parent_dir = '/raid/datasets/FeTS22/MICCAI_FeTS2022_TrainingData' 
    institution_split_csv_filename = 'partitioning_2.csv'

    # start the experiment
    time_start = time.time()

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

    # save experimetn summary to files
    home = str(Path.home())
    scores_dataframe_file = os.path.join(home, '.local/workspace/checkpoint', checkpoint_folder, 'scores_df.csv')
    scores_dataframe.to_csv(scores_dataframe_file)

    time_end = time.time()

    # show the time elapsed for this session
    sesseion_time = np.around((time_end-time_start)/3600, 2)
    print('Session time: {} hrs. That\'s all folks.'.format(sesseion_time))

    time_end_stamp = time.strftime('%y-%m-%d-%H-%M-%S')
    print(f'Session completed at {time_end_stamp}')


    ###################################################################################
    # # Use the following code if you wish to generate inference using this code
    ###################################################################################

    # # you will need to specify the correct experiment folder and the parent directory for
    # # the data you want to run inference over (assumed to be the experiment that just completed)
    # #checkpoint_folder='experiment_1'
    # data_path = '/home/brats/MICCAI_FeTS2022_ValidationData'
    # validation_csv_filename = 'validation.csv'

    # # you can keep these the same if you wish
    # final_model_path = os.path.join(home, '.local/workspace/checkpoint', checkpoint_folder, 'best_model.pkl')

    # # If the experiment is only run for a single round, use the temp model instead
    # if not Path(final_model_path).exists():
    #     final_model_path = os.path.join(home, '.local/workspace/checkpoint', checkpoint_folder, 'temp_model.pkl')
    # outputs_path = os.path.join(home, '.local/workspace/checkpoint', checkpoint_folder, 'model_outputs')

    # # Using this best model, we can now produce NIfTI files for model outputs 
    # # using a provided data directory
    # model_outputs_to_disc(data_path=data_path, 
    #                     validation_csv=validation_csv_filename,
    #                     output_path=outputs_path, 
    #                     native_model_path=final_model_path,
    #                     outputtag='',
    #                     device=device)