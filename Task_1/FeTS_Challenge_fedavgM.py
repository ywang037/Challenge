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
# TODO not finished yet
# def wy_train_col_selector(collaborators,
#                         db_iterator,
#                         fl_round,
#                         collaborators_chosen_each_round,
#                         collaborator_times_per_round):
#     """ Chooses which collaborators will train for a given round:
#         select M out of N from the collaborators of the previous round, as per certain scores, e.g., dist_scores, val_scores, or hybrid_scores
#         then randomly select N-M from the rest.

    
#     Args:
#         collaborators: list of strings of collaborator names
#         db_iterator: iterator over history of all tensors.
#             Columns: ['tensor_name', 'round', 'tags', 'nparray']
#         fl_round: round number
#         collaborators_chosen_each_round: a dictionary of {round: list of collaborators}. Each list indicates which collaborators trained in that given round.
#         collaborator_times_per_round: a dictionary of {round: {collaborator: total_time_taken_in_round}}.  
#     """
#     # logger.info("one_collaborator_on_odd_rounds called!")
#     training_collaborators = []
    
#     # metric_name = 'acc'
#     metric_name = 'loss'
#     tags = ('metric','validate_local')
#     val_loss = {}
#     # all_losses_previous_round = []
#     for record in db_iterator:
#         for col in collaborators_chosen_each_round[fl_round-1]:
#             tags = set(tags + [col])
#             if (
#                 tags <= set(record['tags']) 
#                 and record['round'] == fl_round-1
#                 and record['tensor_name'] == metric_name
#             ):
#                 val_loss[col]=record['nparray']
#                 # all_losses_previous_round.append(record['nparray'])
    
#     # get the index of the sorted loss values
#     all_losses_previous_round = [val_loss[col] for col in collaborators_chosen_each_round(fl_round-1)]
#     sorted_idx = np.argsort(all_losses_previous_round,axis=0)
    
#     # choose 1/3 of the total collaborators from all collaborators participated in the previous round
#     num_col_to_select_from_prevous_round = int(np.round(len(collaborators)/3)) # can be manually set to an integer for partition_1 or partition 2
#     for i in range(num_col_to_select_from_prevous_round):
#         training_collaborators.append(sorted_idx[i])

#     # randomly choose some other collaborators from the rest 
#     num_to_select_from_the_rest = 1 # can be manually set
#     rng = np.random.default_rng(35)
#     the_rest = collaborators - training_collaborators
#     selected_from_the_other = rng.choice(the_rest, num_to_select_from_the_rest, replace=False).tolist()
#     training_collaborators +=selected_from_the_other

#     return training_collaborators

def wy_select_col_with_more_data_1(collaborators,
                                    db_iterator,
                                    fl_round,
                                    collaborators_chosen_each_round,
                                    collaborator_times_per_round):
    # this is a list of ids for collaborator that has at least 10 data samples, hand-picked
    preserved_col_id = [0, 2, 3, 4, 5, 6, 10, 11, 12, 14, 15, 17, 19, 20] # for partition_1
    training_collaborators = [collaborators[i] for i in preserved_col_id]
    
    return training_collaborators

def wy_select_col_with_more_data_2(collaborators,
                                    db_iterator,
                                    fl_round,
                                    collaborators_chosen_each_round,
                                    collaborator_times_per_round):
    # this is a list of ids for collaborator that has at least 10 data samples, hand-picked
    preserved_col_id = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 20, 21, 23, 24, 25, 27, 28, 29, 30] # for partition_2
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
    preserved_col_id = [0, 2, 3, 4, 5, 6, 10, 11, 12, 14, 15, 17, 19, 20] # for partition_1
    
    # hand-calculated problabilities, for partition_1
    prob_of_select = [
        0.4279732, 0.01256281, 
        0.03936348, 0.01842546, 
        0.02847571, 0.01005025, 
        0.01172529, 0.00921273, 
        0.02931323, 0.01088777, 
        0.02512563, 0.319933, 
        0.02763819, 0.02931323
        ]
    prob_of_select = np.array(prob_of_select, dtype=np.float32) # make it a np array

    rng = np.random.default_rng()
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
    preserved_col_id = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 20, 21, 23, 24, 25, 27, 28, 29, 30] # for partition_2
    
    # hand-calculated problabilities, for partition_2
    prob_of_select = [
        0.14237856, 0.14237856, 0.14321608, 0.01256281, 0.01340034, 0.01256281,
        0.01340034, 0.01842546, 0.02847571, 0.01005025, 0.01172529, 0.00921273,
        0.01005025, 0.00921273, 0.01005025, 0.01088777, 0.02512563, 0.10636516,
        0.10636516, 0.10720268, 0.02763819, 0.01005025, 0.00921273, 0.01005025
        ]
    prob_of_select = np.array(prob_of_select, dtype=np.float32) # make it a np array

    rng = np.random.default_rng()
    num_final_pick = 15
    final_picked_id = rng.choice(preserved_col_id, num_final_pick, replace=False, p=prob_of_select)
    training_collaborators = [collaborators[i] for i in final_picked_id]
    
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

def get_val_loss_delta_score(local_tensors,tensor_db,fl_round):
    """ this function gets the reduction in validation loss for each col,
        the reduction is cast to zero if it is negative
        i.e., max(validate_local - validate_agg, 0)
    """
    # metric_name = 'valid_dice'
    metric_name = 'valid_loss'
    tags_local = ('metric','validate_local')
    tags_agg = ('metric','validate_agg')
    val_loss_local, val_loss_agg, val_loss_delta = {}, {}, {}
    for _, record in tensor_db.iterrows():
        for t in local_tensors:
            col = t.col_name
            tags_local = set(tags_local + tuple([col]))
            tags_agg = set(tags_local + tuple([col]))
            record_tags = record['tags']
            if (
                tags_local <= set(record_tags) 
                and record['round'] == fl_round
                and record['tensor_name'] == metric_name
            ):
                val_loss_local[col]=record['nparray']
            if (
                tags_agg <= set(record_tags) 
                and record['round'] == fl_round
                and record['tensor_name'] == metric_name
            ):
                val_loss_agg[col]=record['nparray']
                val_loss_delta[col] = max(val_loss_local - val_loss_agg[col],0) # local validation is supposed to be done after agg model validation

    sum=0
    for _, loss in val_loss_delta.items():
        sum += loss
    if sum:        
        # if sum is not zero, i.e., at least one col has positive reduction in val loss, then do normalization
        for col, loss in val_loss_delta.items():
            val_loss_delta[col] = loss/sum   
        val_loss_delta_scores = np.array([val_loss_delta[t.col_name] for t in local_tensors],dtype=np.float64)
    else:
        # if none of the cols reduces local validation loss, then output None, 
        # then let the caller function decide if output is None
        val_loss_delta = None
        val_loss_delta_scores = None
    
    return val_loss_delta, val_loss_delta_scores

def get_hybrid_score_val(local_tensors, tensor_db, tensor_name, fl_round):
    # get the scores w.r.t. to closseness/distance
    dist_scores = get_dist_score(local_tensors, tensor_db, tensor_name, fl_round)
    
    # get the scores w.r.t. local validation loss
    _, val_loss_scores = get_val_loss_score(local_tensors,tensor_db,fl_round)

    # get the hybrid scores
    hybrid_scores = dist_scores * val_loss_scores

    # normalize the score into [0 ,1]
    hybrid_scores = hybrid_scores/hybrid_scores.sum()

    return hybrid_scores

def get_hybrid_score_val_delta(local_tensors, tensor_db, tensor_name, fl_round):
    # get the scores w.r.t. to closseness/distance
    dist_scores = get_dist_score(local_tensors, tensor_db, tensor_name, fl_round)
    
    # get the scores w.r.t. local validation loss
    # _, val_loss_scores = get_val_loss_delta_score(local_tensors,tensor_db,fl_round)
    _, val_delta_scores = get_val_loss_delta_score(local_tensors,tensor_db,fl_round)

    # get the hybrid scores
    if val_delta_scores is not None:
        hybrid_scores = dist_scores * val_delta_scores
    else:
        # if none of the cols reduces local validation loss, then just use dist-based scores instead
        hybrid_scores = dist_scores

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

# # the following method is depracted
# def wy_agg_func_dist2(local_tensors,
#                         tensor_db,
#                         tensor_name,
#                         fl_round,
#                         collaborators_chosen_each_round,
#                         collaborator_times_per_round):
#     """ this aggregation function finds the aggregated model by weighting local model updates (relative to the previous round) with the normliazed distance 
#         the distance is measured from each local update to the centroid of all local model udpates
#     """
    
#     if fl_round == 0:
#         # in the first round, just do normal fedavg
#         tensor_values = [t.tensor for t in local_tensors]
#         weight_values = [t.weight for t in local_tensors]               
#         new_tensor_agg =  np.average(tensor_values, weights=weight_values, axis=0)   
        
#         # besides, also save the aggregated model for the computing of model update direction in the later roudns [optional]
#         tensor_db.store(tensor_name=tensor_name, tags=('tensor_agg_round_0',), nparray=new_tensor_agg)
#     else:
#         # get the scores w.r.t. to closseness/distance
#         dist_scores = get_dist_score2(local_tensors, tensor_db, tensor_name, fl_round)
        
#         # compute the aggregated model
#         tensor_values = [t.tensor for t in local_tensors]
#         weight_values = [dist_scores[t.col_name] for t in local_tensors]
#         new_tensor_agg = np.average(tensor_values, weights=weight_values, axis=0)  
    
#     return new_tensor_agg

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

def wy_agg_func_val_delta(local_tensors,
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
        val_loss_delta, _ = get_val_loss_delta_score(local_tensors,tensor_db,fl_round)
        
        # compute the aggregated model
        tensor_values = [t.tensor for t in local_tensors]
        if val_loss_delta is not None:
            weight_values = [val_loss_delta[t.col_name] for t in local_tensors]
        else:
            # if none of cols reduces local validation loss, just use standard fedavg weights
            weight_values = [t.weight for t in local_tensors]    
        new_tensor_agg = np.average(tensor_values, weights=weight_values, axis=0)  
    
    return new_tensor_agg

def wy_agg_func_hybrid_val(local_tensors,
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
        weight_values = get_hybrid_score_val(local_tensors, tensor_db, tensor_name, fl_round)

        # compute the aggregated model using the distance score directly
        tensor_values = [t.tensor for t in local_tensors]
        new_tensor_agg = np.average(tensor_values, weights=weight_values, axis=0)  
    
    return new_tensor_agg

def wy_agg_func_hybrid_val_delta(local_tensors,
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
        weight_values = get_hybrid_score_val_delta(local_tensors, tensor_db, tensor_name, fl_round)

        # compute the aggregated model using the distance score directly
        tensor_values = [t.tensor for t in local_tensors]
        new_tensor_agg = np.average(tensor_values, weights=weight_values, axis=0)  
    
    return new_tensor_agg


def wy_agg_func_dist_adv(local_tensors,
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
        dist_scores = get_dist_score(local_tensors, tensor_db, tensor_name, fl_round) 

        # compute the final scores
        final_scores = weight_values_default * dist_scores

        # normalize the score into [0 ,1]
        weight_values = final_scores/final_scores.sum()
        
        # compute the aggregated model
        tensor_values = [t.tensor for t in local_tensors]
        new_tensor_agg = np.average(tensor_values, weights=weight_values, axis=0)

    return new_tensor_agg

def wy_agg_func_dist_adv2(local_tensors,
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
        dist_scores = get_dist_score(local_tensors, tensor_db, tensor_name, fl_round) 

        # compute the final scores
        alpha = np.float64(0.5)
        final_scores = alpha*weight_values_default + (1-alpha)*dist_scores

        # normalize the score into [0 ,1]
        weight_values = final_scores/final_scores.sum()

        # then find the final output as the weighted average of the above two aggregated models       
        new_tensor_agg=  np.average(tensor_values, weights=weight_values, axis=0)

    return new_tensor_agg

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

def wy_agg_func_val_adv2(local_tensors,
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
        _, val_loss_scores = get_val_loss_score(local_tensors,tensor_db,fl_round)

        # compute the final scores
        alpha = np.float64(0.5)
        final_scores = alpha*weight_values_default + (1-alpha)*val_loss_scores

        # normalize the score into [0 ,1]
        weight_values = final_scores/final_scores.sum()

        # then find the final output as the weighted average of the above two aggregated models       
        new_tensor_agg=  np.average(tensor_values, weights=weight_values, axis=0)

    return new_tensor_agg

def wy_agg_func_val_delta_adv(local_tensors,
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
        _, val_delta_scores = get_val_loss_delta_score(local_tensors,tensor_db,fl_round)

        # compute the final scores
        if val_delta_scores is not None:
            final_scores = weight_values_default * val_delta_scores
        else:
            final_scores = weight_values_default

        # normalize the score into [0 ,1]
        weight_values = final_scores/final_scores.sum()
        
        # compute the aggregated model
        tensor_values = [t.tensor for t in local_tensors]
        new_tensor_agg = np.average(tensor_values, weights=weight_values, axis=0)

    return new_tensor_agg

def wy_agg_func_val_delta_adv2(local_tensors,
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
        _, val_delta_scores = get_val_loss_delta_score(local_tensors,tensor_db,fl_round)

        # compute the final scores
        alpha = np.float64(0.5)
        if val_delta_scores is not None:
            final_scores = alpha*weight_values_default + (1-alpha)*val_delta_scores
        else:
            final_scores = weight_values_default
        # normalize the score into [0 ,1]
        weight_values = final_scores/final_scores.sum()

        # then find the final output as the weighted average of the above two aggregated models       
        new_tensor_agg=  np.average(tensor_values, weights=weight_values, axis=0)

    return new_tensor_agg

def wy_agg_func_hybrid_val_adv(local_tensors,
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
        hybrid_val_scores = get_hybrid_score_val(local_tensors,tensor_db,fl_round)

        # compute the final scores
        final_scores = weight_values_default * hybrid_val_scores

        # normalize the score into [0 ,1]
        weight_values = final_scores/final_scores.sum()
        
        # compute the aggregated model
        tensor_values = [t.tensor for t in local_tensors]
        new_tensor_agg = np.average(tensor_values, weights=weight_values, axis=0)

    return new_tensor_agg

def wy_agg_func_hybrid_val_adv2(local_tensors,
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
        hybrid_val_scores = get_hybrid_score_val(local_tensors,tensor_db,fl_round)

        # compute the final scores
        alpha = np.float64(0.5)
        final_scores = alpha*weight_values_default + (1-alpha)*hybrid_val_scores

        # normalize the score into [0 ,1]
        weight_values = final_scores/final_scores.sum()

        # then find the final output as the weighted average of the above two aggregated models       
        new_tensor_agg=  np.average(tensor_values, weights=weight_values, axis=0)

    return new_tensor_agg

def wy_agg_func_hybrid_val_delta_adv(local_tensors,
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
        hybrid_val_delta_scores = get_hybrid_score_val_delta(local_tensors,tensor_db,fl_round)

        # compute the final scores
        final_scores = weight_values_default * hybrid_val_delta_scores

        # normalize the score into [0 ,1]
        weight_values = final_scores/final_scores.sum()
        
        # compute the aggregated model
        tensor_values = [t.tensor for t in local_tensors]
        new_tensor_agg = np.average(tensor_values, weights=weight_values, axis=0)

    return new_tensor_agg

def wy_agg_func_hybrid_val_delta_adv2(local_tensors,
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
        hybrid_val_delta_scores = get_hybrid_score_val_delta(local_tensors,tensor_db,fl_round)

        # compute the final scores
        alpha = np.float64(0.5)
        final_scores = alpha*weight_values_default + (1-alpha)*hybrid_val_delta_scores

        # normalize the score into [0 ,1]
        weight_values = final_scores/final_scores.sum()

        # then find the final output as the weighted average of the above two aggregated models       
        new_tensor_agg=  np.average(tensor_values, weights=weight_values, axis=0)

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

# define centralized-alike approach
def central_train_sim(local_tensors,
                    tensor_db,
                    tensor_name,
                    fl_round,
                    collaborators_chosen_each_round,
                    collaborator_times_per_round):
    """ in this case, partition_0.csv is to be used, only one collaborator 'col 1' who has all the training data
        to simulate a centralized training
    """
    tensor_values = local_tensors[0].tensor # the only one 
    return np.average(tensor_values, axis=0)

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
aggregation_function = FedAvgM_Selection
# aggregation_function = fedNova_simplified
# aggregation_function = central_train_sim

# # choose from the following list for customized aggregation function
# # dist-based methods
# aggregation_function = wy_agg_func_dist # plain
# aggregation_function = wy_agg_func_dist_adv # adv
# aggregation_function = wy_agg_func_dist_adv2 # adv2

# # val loss based methods
# aggregation_function = wy_agg_func_val # plain
# aggregation_function = wy_agg_func_val_adv # adv
# aggregation_function = wy_agg_func_val_adv2 # adv2

# # delta val loss based methods
# aggregation_function = wy_agg_func_val_delta # plain
# aggregation_function = wy_agg_func_val_delta_adv # adv
# aggregation_function = wy_agg_func_val_delta_adv2 # adv2

# # val & dist hybrid methods
# aggregation_function = wy_agg_func_hybrid_val # basic
# aggregation_function = wy_agg_func_hybrid_val_adv # adv
# aggregation_function = wy_agg_func_hybrid_val_adv2 # adv2

# # val & dist hybrid methods
# aggregation_function = wy_agg_func_hybrid_val_delta # basic 
# aggregation_function = wy_agg_func_hybrid_val_delta_adv # adv
# aggregation_function = wy_agg_func_hybrid_val_delta_adv2 # adv2

# training col selection strategy
choose_training_collaborators = all_collaborators_train

# hyper param.
training_hyper_parameters_for_round = constant_hyper_parameters

# As mentioned in the 'Custom Aggregation Functions' section (above), six 
# perfomance evaluation metrics are included by default for validation outputs in addition 
# to those you specify immediately above. Changing the below value to False will change 
# this fact, excluding the three hausdorff measurements. As hausdorff distance is 
# expensive to compute, excluding them will speed up your experiments.
include_validation_with_hausdorff=False

# We encourage participants to experiment with partitioning_1 and partitioning_2, as well as to create
# other partitionings to test your changes for generalization to multiple partitionings.
institution_split_csv_filename = 'small_split.csv'
# institution_split_csv_filename = 'partitioning_1.csv'
# institution_split_csv_filename = 'partitioning_2.csv'
# institution_split_csv_filename = 'partitioning_2_top5_clients.csv'
# institution_split_csv_filename = 'partitioning_2_rand_pick_5.csv'


# change this to point to the parent directory of the data
brats_training_data_parent_dir = '/mnt/data/home/wangyuan/Challenge/Data/TrainingData'

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
data_path = '/mnt/data/home/wangyuan/Challenge/Data/ValidationData'
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

time_end_stamp = time.strftime('%y-%m-%d-%H-%M-%S')
print(f'Session completed at {time_end_stamp}')