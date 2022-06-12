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


##########################################################
# # Custom selection functions - WY's trials
##########################################################
def wy_select_col_with_more_data_1(collaborators,
                                    db_iterator,
                                    fl_round,
                                    collaborators_chosen_each_round,
                                    collaborator_times_per_round):
    # this is a list of ids for collaborator that has at least 10 data samples, hand-picked
    preserved_col_id = [0, 2, 3, 4, 5, 6, 10, 11, 12, 14, 15, 17, 19, 20, 21] # for partition_1
    training_collaborators = [collaborators[i] for i in preserved_col_id]
    
    return training_collaborators

def wy_select_col_with_more_data_2(collaborators,
                                    db_iterator,
                                    fl_round,
                                    collaborators_chosen_each_round,
                                    collaborator_times_per_round):
    # this is a list of ids for collaborator that has at least 10 data samples, hand-picked
    preserved_col_id = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 20, 21, 23, 24, 25, 27, 28, 29, 30, 31] # for partition_2
    training_collaborators = [collaborators[i] for i in preserved_col_id]
    
    return training_collaborators

def wy_select_col_with_more_data_subset_1(collaborators,
                                        db_iterator,
                                        fl_round,
                                        collaborators_chosen_each_round,
                                        collaborator_times_per_round):
    """ this function randomly select a subset of collaborators from the hand-picked list of collaborators with more than 10 data samples
        the random selection using the probabilities which are normalized number of training samples 
    """
    # this is a list of ids for collaborator that has at least 10 data samples, hand-picked
    preserved_col_id = [0, 2, 3, 4, 5, 6, 10, 11, 12, 14, 15, 17, 19, 20, 21] # for partition_1
    
    # hand-calculated problabilities, for partition_1
    prob_of_select = [
        0.42371476, 0.01243781, 0.03897181, 0.01824212, 0.02819237, 
        0.00995025, 0.01160862, 0.00912106, 0.02902156, 0.01077944, 
        0.02487562, 0.31674959, 0.02736318, 0.02902156, 0.00995025
        ]
    prob_of_select = np.array(prob_of_select, dtype=np.float32) # make it a np array

    rng = np.random.default_rng(35)
    num_final_pick = 10
    final_picked_id = rng.choice(preserved_col_id, num_final_pick, replace=False, p=prob_of_select)
    training_collaborators = [collaborators[i] for i in final_picked_id]

    return training_collaborators

def wy_select_col_with_more_data_subset_2(collaborators,
                                        db_iterator,
                                        fl_round,
                                        collaborators_chosen_each_round,
                                        collaborator_times_per_round):
    """ this function randomly select a subset of collaborators from the hand-picked list of collaborators with more than 10 data samples
        the random selection using the probabilities which are normalized number of training samples 
    """
    
    # this is a list of ids for collaborator that has at least 10 data samples, hand-picked
    preserved_col_id = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 20, 21, 23, 24, 25, 27, 28, 29, 30, 31] # for partition_2
    
    # hand-calculated problabilities, for partition_2
    prob_of_select = [
        0.14096186, 0.14096186, 0.14179104, 0.01243781, 0.013267,
        0.01243781, 0.013267  , 0.01824212, 0.02819237, 0.00995025,
        0.01160862, 0.00912106, 0.00995025, 0.00912106, 0.00995025,
        0.01077944, 0.02487562, 0.1053068 , 0.1053068 , 0.10613599,
        0.02736318, 0.00995025, 0.00912106, 0.00995025, 0.00995025
        ]
    prob_of_select = np.array(prob_of_select, dtype=np.float32) # make it a np array

    rng = np.random.default_rng(35)
    num_final_pick = 15
    final_picked_id = rng.choice(preserved_col_id, num_final_pick, replace=False, p=prob_of_select)
    training_collaborators = [collaborators[i] for i in final_picked_id]
    
    return training_collaborators



# # Custom hyperparameters for training

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



##########################################################
# # Custom Aggregation Functions - WY's trials
##########################################################
def find_previous_tensor_agg(tensor_db, tensor_name, fl_round):
    previous_tensor_value = None
    for _, record in tensor_db.iterrows():
        if (
            # record['round'] == (fl_round - 1)
            record['round'] == fl_round
            and record['tensor_name'] == tensor_name
            and 'aggregated' in record['tags']
            and 'delta' not in record['tags']
            ):
            previous_tensor_value = record['nparray']
            break
    
    # # another way of getting the aggregated model of the previous round
    # previous_tensor_value = tensor_db.search(tensor_name=tensor_name, fl_round=fl_round, tags=('model',), origin='aggregator')
    # previous_tensor_value = previous_tensor_value.nparray.iloc[0]
    return previous_tensor_value

def get_dist_score(local_tensors, tensor_db, tensor_name, fl_round):
    """ compute the scores for each local model updates which is inverse propotional to the euclidean distance from the local models to the centroid
    """
    
    # get the aggregated model parameters of the previous round
    previous_tensor_agg = find_previous_tensor_agg(tensor_db, tensor_name, fl_round) # using the aggregated model of the previous round as the reference point
    # privious_tensor_agg = tensor_db.retrieve(tensor_name=tensor_name, tags=('tensor_agg_round_0',)) # [optional] using the aggregated model of first round as the reference point
    
    assert previous_tensor_agg is not None, 'ERROR: previous aggregated tensor is None'

    # compute the model updates of each col
    deltas = [t.tensor - previous_tensor_agg for t in local_tensors]

    # compute the centroid of the model updates as the mean and/or median
    deltas_cent = np.mean(deltas, axis=0)
    # deltas_cent = np.median(deltas, axis=0)
    
    # compute the euclidean distance to the centroid of model udpates 
    deltas_norm = [np.linalg.norm(d-deltas_cent) for d in deltas]
    deltas_norm = np.array(deltas_norm, dtype=np.float64)

    # map the euclidean distance with exponential function to make the closeness score inverse propotional to the distance
    gamma = np.float64(1e-2)
    dist_scores = np.exp(-gamma*deltas_norm)

    # normalize the distance score into [0 ,1]
    dist_scores = dist_scores/dist_scores.sum()
    
    return dist_scores

def get_dist_score2(local_tensors, tensor_db, tensor_name, fl_round):
    """ compute the scores for each local model updates which is inverse propotional to the euclidean distance from the local models to the centroid
    """
    gamma = np.float64(1e-2)

    # get the aggregated model parameters of the previous round
    previous_tensor_agg = find_previous_tensor_agg(tensor_db, tensor_name, fl_round) # using the aggregated model of the previous round as the reference point
    # privious_tensor_agg = tensor_db.retrieve(tensor_name=tensor_name, tags=('tensor_agg_round_0',)) # [optional] using the aggregated model of first round as the reference point
    
    assert previous_tensor_agg is not None, 'ERROR: previous aggregated tensor is None'

    # compute the model updates of each col
    deltas={}
    dist_scores={}
    for t in local_tensors:
        deltas[t.col_name] = t.tensor - previous_tensor_agg

    # compute the centroid of the model updates as the mean and/or median
    deltas_cent = np.mean([tensor for _, tensor in deltas.items()], axis=0)
    # deltas_cent = np.median(deltas, axis=0)
    
    # compute the euclidean distance to the centroid of model udpates 
    # map the euclidean distance with exponential function to make the closeness score inverse propotional to the distance
    sum=0
    for col, tensor in deltas.items():
        score = np.exp(-gamma*np.linalg.norm(tensor-deltas_cent)) 
        dist_scores[col] = score
        sum += score
    
    # normalize the score to [0 ,1]
    for col, score in dist_scores.items():
        dist_scores[col] = dist_scores[col]/sum

    return dist_scores

# get the validation loss of every collaborators in a specific round
def get_val_loss_score(local_tensors,tensor_db,fl_round):
    # metric_name = 'acc'
    metric_name = 'loss'
    tags = ('metric','validate_local')
    val_loss = {}
    for record in tensor_db.iterrows():
        for t in local_tensors:
            # tags = set(tags + tuple([t.col_name]))
            some_tuple = record['tags']
            if (
                set(tags + tuple([t.col_name])) <= set(some_tuple) 
                and record['round'] == fl_round-1
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
    # val_loss = []
    # for _, record in tensor_db.iterrows():
    #     for local_tensor in local_tensors:
    #         tags = set(tags + tuple([local_tensor.col_name]))
    #         if (
    #             tags <= set(record['tags']) 
    #             and record['round'] == fl_round
    #             and record['tensor_name'] == metric_name
    #         ):
    #             val_loss.append(record['nparray'])
    # return np.array(val_loss,dtype=np.float64)

def get_hybrid_score(local_tensors, tensor_db, tensor_name, fl_round):
    # get the scores w.r.t. to closseness/distance
    dist_scores = get_dist_score(local_tensors, tensor_db, tensor_name, fl_round)
    
    # get the scores w.r.t. local validation loss
    _, val_scores = get_val_loss_score(local_tensors,tensor_db,fl_round)

    # get the hybrid scores
    hybrid_scores = dist_scores * val_scores

    # normalize the score into [0 ,1]
    hybrid_scores = hybrid_scores/hybrid_scores.sum()

    return hybrid_scores

def wy_agg_func_dist(local_tensors,
                    tensor_db,
                    tensor_name,
                    fl_round,
                    collaborators_chosen_each_round,
                    collaborator_times_per_round):
    """ this aggregation function finds the aggregated model by weighting local model updates (relative to the previous round) with the normliazed distance 
        the distance is measured from each local update to the centroid of all local model udpates
    """
    
    if fl_round == 0:
        # in the first round, just do normal fedavg
        tensor_values = [t.tensor for t in local_tensors]
        weight_values = [t.weight for t in local_tensors]               
        new_tensor_agg = np.average(tensor_values, weights=weight_values, axis=0)   
        
        # besides, also save the aggregated model for the computing of model update direction in the later roudns [optional]
        tensor_db.store(tensor_name=tensor_name, tags=('tensor_agg_round_0',), nparray=new_tensor_agg)
    else:
        # weighting by the scores w.r.t. to closseness/distance
        weight_values = get_dist_score(local_tensors, tensor_db, tensor_name, fl_round)

        # compute the aggregated model using the distance score directly
        tensor_values = [t.tensor for t in local_tensors]
        new_tensor_agg = np.average(tensor_values, weights=weight_values, axis=0)  
    
    return new_tensor_agg

def wy_agg_func_dist2(local_tensors,
                        tensor_db,
                        tensor_name,
                        fl_round,
                        collaborators_chosen_each_round,
                        collaborator_times_per_round):
    """ this aggregation function finds the aggregated model by weighting local model updates (relative to the previous round) with the normliazed distance 
        the distance is measured from each local update to the centroid of all local model udpates
    """
    
    if fl_round == 0:
        # in the first round, just do normal fedavg
        tensor_values = [t.tensor for t in local_tensors]
        weight_values = [t.weight for t in local_tensors]               
        new_tensor_agg =  np.average(tensor_values, weights=weight_values, axis=0)   
        
        # besides, also save the aggregated model for the computing of model update direction in the later roudns [optional]
        tensor_db.store(tensor_name=tensor_name, tags=('tensor_agg_round_0',), nparray=new_tensor_agg)
    else:
        # get the scores w.r.t. to closseness/distance
        dist_scores = get_dist_score2(local_tensors, tensor_db, tensor_name, fl_round)
        
        # compute the aggregated model
        tensor_values = [t.tensor for t in local_tensors]
        weight_values = [dist_scores[t.col_name] for t in local_tensors]
        new_tensor_agg = np.average(tensor_values, weights=weight_values, axis=0)  
    
    return new_tensor_agg

def wy_agg_func_val(local_tensors,
                    tensor_db,
                    tensor_name,
                    fl_round,
                    collaborators_chosen_each_round,
                    collaborator_times_per_round):
    """ this aggregation function finds the aggregated model by weighting local model updates (relative to the previous round) with the local validation losses
    """
    
    if fl_round == 0:
        # in the first round, just do normal fedavg
        tensor_values = [t.tensor for t in local_tensors]
        weight_values = [t.weight for t in local_tensors]               
        new_tensor_agg = np.average(tensor_values, weights=weight_values, axis=0)   
        
        # besides, also save the aggregated model for the computing of model update direction in the later roudns [optional]
        tensor_db.store(tensor_name=tensor_name, tags=('tensor_agg_round_0',), nparray=new_tensor_agg)
    else:        
        # get the scores w.r.t. local validation loss
        val_loss, _ = get_val_loss_score(local_tensors,tensor_db,fl_round)
        
        # compute the aggregated model
        tensor_values = [t.tensor for t in local_tensors]
        weight_values = [val_loss[t.col_name] for t in local_tensors]
        new_tensor_agg = np.average(tensor_values, weights=weight_values, axis=0)  
    
    return new_tensor_agg

# as the wy_agg_func_val is not working, this function is pending revision
def wy_agg_func_hybrid(local_tensors,
                        tensor_db,
                        tensor_name,
                        fl_round,
                        collaborators_chosen_each_round,
                        collaborator_times_per_round):
    """ this aggregation function finds the aggregated model by weighting local model updates (relative to the previous round) 
        with the normalized products of the distance scores and validation scores
        the hybrid score is similar to eqn (3) in https://arxiv.org/abs/2012.08565
    """
    
    if fl_round == 0:
        # in the first round, just do normal fedavg
        tensor_values = [t.tensor for t in local_tensors]
        weight_values = [t.weight for t in local_tensors]               
        new_tensor_agg = np.average(tensor_values, weights=weight_values, axis=0)   
        
        # besides, also save the aggregated model for the computing of model update direction in the later roudns [optional]
        tensor_db.store(tensor_name=tensor_name, tags=('tensor_agg_round_0',), nparray=new_tensor_agg)
    else:        
        # weighting by the hybrid scores
        weight_values = get_hybrid_score(local_tensors, tensor_db, tensor_name, fl_round)

        # compute the aggregated model using the distance score directly
        tensor_values = [t.tensor for t in local_tensors]
        new_tensor_agg = np.average(tensor_values, weights=weight_values, axis=0)  
    
    return new_tensor_agg

def wy_agg_func_adv(local_tensors,
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
        first_scores = get_dist_score(local_tensors, tensor_db, tensor_name, fl_round) 
        # _, first_scores = get_val_loss_score(local_tensors,tensor_db,fl_round)
        # first_scores = get_hybrid_score(local_tensors, tensor_db, tensor_name, fl_round)

        # compute the final scores
        final_scores = weight_values_default * first_scores

        # normalize the score into [0 ,1]
        weight_values = final_scores/final_scores.sum()
        
        # compute the aggregated model
        tensor_values = [t.tensor for t in local_tensors]
        new_tensor_agg = np.average(tensor_values, weights=weight_values, axis=0)

    return new_tensor_agg

def wy_agg_func_adv2(local_tensors,
                    tensor_db,
                    tensor_name,
                    fl_round,
                    collaborators_chosen_each_round,
                    collaborator_times_per_round):
    """ advanced aggregation, which is a further weighted average of fedavg and customized aggregation function
        this function share the simimiar spirit of eqn (2) in https://arxiv.org/abs/2111.08649 (the winner of FeTS2021)
    """
    if fl_round == 0:
        # in the first round, just do normal fedavg
        tensor_values = [t.tensor for t in local_tensors]
        weight_values = [t.weight for t in local_tensors]               
        new_tensor_agg =  np.average(tensor_values, weights=weight_values, axis=0)   
        
        # besides, also save the aggregated model for the computing of model update direction in the later roudns [optional]
        tensor_db.store(tensor_name=tensor_name, tags=('tensor_agg_round_0',), nparray=new_tensor_agg)
    else:
        # get the tensors to be aggregated
        tensor_values = [t.tensor for t in local_tensors]
        
        # get the defualt weights values (shall be like weighted by number of training samples)
        weight_values_default = np.array([t.weight for t in local_tensors], dtype=np.float64)               

        # compute the customized aggregation weights, i.e., scores associated to each local tensor
        # these scores can be the output of get_dist_score, or get_val_loss, or get_hybrid_score
        first_scores = get_dist_score(local_tensors, tensor_db, tensor_name, fl_round) 
        # _, first_scores = get_val_loss_score(local_tensors,tensor_db,fl_round)
        # first_scores = get_hybrid_score(local_tensors, tensor_db, tensor_name, fl_round)

        # compute the final scores
        alpha = np.float64(0.5)
        final_scores = alpha*weight_values_default + (1-alpha)*first_scores

        # normalize the score into [0 ,1]
        weight_values = final_scores/final_scores.sum()

        # then find the final output as the weighted average of the above two aggregated models       
        new_tensor_agg=  np.average(tensor_values, weights=weight_values, axis=0)

    return new_tensor_agg


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
             db_iterator,
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
        for record in db_iterator:
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
# aggregation_function = weighted_average_aggregation
# aggregation_function = FedAvgM_Selection
# aggregation_function = wy_agg_func_dist2
# aggregation_function = wy_agg_func_dist
# aggregation_function = wy_agg_func_val
# aggregation_function = wy_agg_func_hybrid
# aggregation_function = wy_agg_func_adv
aggregation_function = wy_agg_func_adv2

# choose_training_collaborators = all_collaborators_train
choose_training_collaborators = wy_select_col_with_more_data_1
choose_training_collaborators = wy_select_col_with_more_data_2
choose_training_collaborators = wy_select_col_with_more_data_subset_1
choose_training_collaborators = wy_select_col_with_more_data_subset_2

training_hyper_parameters_for_round = constant_hyper_parameters

# As mentioned in the 'Custom Aggregation Functions' section (above), six 
# perfomance evaluation metrics are included by default for validation outputs in addition 
# to those you specify immediately above. Changing the below value to False will change 
# this fact, excluding the three hausdorff measurements. As hausdorff distance is 
# expensive to compute, excluding them will speed up your experiments.
include_validation_with_hausdorff=True

# We encourage participants to experiment with partitioning_1 and partitioning_2, as well as to create
# other partitionings to test your changes for generalization to multiple partitionings.
institution_split_csv_filename = 'partitioning_2.csv'
# institution_split_csv_filename = 'small_split.csv'

# change this to point to the parent directory of the data
brats_training_data_parent_dir = '/home/wang_yuan/fets2022/Data/TrainingData'

# increase this if you need a longer history for your algorithms
# decrease this if you need to reduce system RAM consumption
db_store_rounds = 10

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


# ## Produce NIfTI files for best model outputs on the validation set
# Now we will produce model outputs to submit to the leader board.
# 
# At the end of every experiment, the best model (according to average ET, TC, WT DICE) 
# is saved to disk at: ~/.local/workspace/checkpoint/\<checkpoint folder\>/best_model.pkl,
# where \<checkpoint folder\> is the one printed to stdout during the start of the 
# experiment (look for the log entry: "Created experiment folder experiment_##..." above).


from fets_challenge import model_outputs_to_disc
from pathlib import Path

# infer participant home folder
home = str(Path.home())

# you will need to specify the correct experiment folder and the parent directory for
# the data you want to run inference over (assumed to be the experiment that just completed)

#checkpoint_folder='experiment_1'
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

time_end = time.time()

# show the time elapsed for this session
sesseion_time = np.around((time_end-time_start)/3600, 2)
print('Session time: {} hrs. That\'s all folks.'.format(sesseion_time))
