import os
from argparse import Namespace
from collections import Counter
import json
import re
import string

import numpy as np
from numpy.core.defchararray import index
from numpy.lib.function_base import append
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import json
from typing import Awaitable, Tuple
from qwikidata.sparql import (get_subclasses_of_item,
                              return_sparql_query_results)

import construction as cons
import funcs

import math
import csv
import random

ROOT_ID = "Q183"

def load_glove_from_file(glove_filepath):
    """
    Load the GloVe embeddings 
    
    Args:
        glove_filepath (str): path to the glove embeddings file 
    Returns:
        word_to_index (dict), embeddings (numpy.ndarary)
    """

    word_to_index = {}
    embeddings = []
    with open(glove_filepath, "r", encoding='utf-8') as fp:
        for index, line in enumerate(fp):
            line = line.split(" ") # each line: word num1 num2 ...
            word_to_index[line[0]] = index # word = line[0] 
            embedding_i = np.array([float(val) for val in line[1:]])
            embeddings.append(embedding_i)
    return word_to_index, np.stack(embeddings)

def make_embedding_matrix(glove_filepath, entity_list):
    """
    Create embedding matrix for a specific set of words.
    
    Args:
        glove_filepath (str): file path to the glove embeddigns
        words (list): list of words in the dataset
    """
    word_to_idx, glove_embeddings = load_glove_from_file(glove_filepath)
    embedding_size = glove_embeddings.shape[1]
    
    final_embeddings = np.zeros((len(entity_list), embedding_size))
    print("There are " + str(len(entity_list)) + " entities")

    word_count = 0
    final_word_to_idx = {}
    matching_dic = {}

    default_embedding = glove_embeddings[word_to_idx["region"]]

    for index, one_entity_item in enumerate(entity_list):
        find_result = funcs.get_proper_embed_index(one_entity_item[1], word_to_idx)
        index_list = find_result[0]
        word_list = find_result[1]
        matching_dic[one_entity_item[0]] = word_list # adjusted to: Qxxx: words.
        average_glove_embeddings = np.zeros(embedding_size) #??
        #print("index: " + str(index))
        #print("word list are: " + str(word_list))
        #print("index list are: " + str(index_list))
        for candidate_index in index_list:
            if(candidate_index < 0):
                average_glove_embeddings = average_glove_embeddings + default_embedding
            else:
                average_glove_embeddings = average_glove_embeddings + glove_embeddings[candidate_index]

        for i in range(embedding_size):
            average_glove_embeddings[i] = average_glove_embeddings[i] / len(index_list) #????
 
        final_embeddings[word_count, :] = average_glove_embeddings
        '''
            torch.nn.init.xavier_uniform_(embedding_i)
        '''
        final_word_to_idx[one_entity_item[1]] = word_count
        word_count = word_count + 1
        
    print("Matching dic size: " + str(len(matching_dic)))
    return final_word_to_idx, final_embeddings, matching_dic


def get_embeddings(parameters, entity_list = []):
    if(parameters["embedding_from_file"] == True):
        print("Load embeddings and word_dic from file")
        with open(parameters["word_to_idx_path"],'r') as f:
            word_to_idx = json.load(f)
        df_embeddings = pd.read_csv(parameters["from_embedding_path"])
        embeddings = np.array(df_embeddings)
        with open(parameters["match_path"],'r') as f:
            matching_dic = json.load(f)
        return word_to_idx, embeddings, matching_dic
    else:
        print("Get new embeddings & dic")
        word_to_idx, embeddings, matching_dic = make_embedding_matrix(parameters["glove_filepath"], entity_list)
        funcs.embedding_to_csv(embeddings, parameters["final_embedding_path"])
        funcs.word_dic_to_json(word_to_idx, parameters["word_to_idx_path"])
        funcs.match_dic_to_json(matching_dic, parameters["match_path"])
        funcs.print_matching_rate(matching_dic)
        return word_to_idx, embeddings, matching_dic


def generate_childrendata(root):    #return a children dic {entity_id: label}
    child_dic = {}
    children_data = {}
    for key, value in root.items():
        child_dic = value["subdivision"]
    for key, value in child_dic.items():
        children_data[key] = value["label"]
    return children_data

def get_neighbor_query(wd_id):
    part1 = """
    SELECT DISTINCT ?subdivision ?subdivisionLabel 
    WHERE
    {
    """
    part2 = "wd:" + wd_id + " wdt:P47 ?subdivision ."
    part3 = """
    ?subdivision rdfs:label ?subdivisionLabel .
    FILTER(lang(?subdivisionLabel)='en')
    FILTER NOT EXISTS{?subdivision wdt:P582|wdt:P576 ?end}
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }

    }
    ORDER BY ?subdivisionLabel
    """
    return part1 + part2 + part3

def get_neighbor_list(entity_label, entity_id):  #finished, return: [entity_id, entity_label, nei_entity_id, nei_entity_label]
    sparql_query = get_neighbor_query(entity_id)
    nei_relation_list = []
    list_item = []
    error = False
    count = 0
    while(error == False and count < 10):
        try:
            res = return_sparql_query_results(sparql_query)
        except BaseException:
            pass
        else:
            if(len(res["results"]["bindings"]) != 0):
                neighbors = res["results"]["bindings"]
                for neighbor in neighbors:
                    nei_entity_label = neighbor["subdivisionLabel"]["value"]
                    nei_entity_id = neighbor["subdivision"]["value"].split('/')[-1]
                    list_item = [entity_id, entity_label, nei_entity_id, nei_entity_label]
                    nei_relation_list.append(list_item)
                error = True
            else:
                count = count + 1 #In fact no "shared border with"
    return nei_relation_list

####################################################################################################################

class Network(nn.Module):
    def __init__(self, center_size, entity_size, embeddings):
        """
        Args:
            center_size (int): size of the center vectors
            entity_size (int): number of entities(embed)

        the center vector matrix is considered as the parameter
        """
        super(Network, self).__init__()

        embeddings = torch.from_numpy(embeddings).float()
        self.in_emb = nn.Embedding(num_embeddings=entity_size,
                                embedding_dim=center_size,
                                _weight=embeddings)
        self.in_emb.weight.requires_grad = True
        #from one-hot to center-vector

    def forward(self, index_vec, neighbor_index_vec, non_neighbor1, non_neighbor2, radius_sum, radius_sum2, radius_sum3, exist_non_neighbor): #input: one-hot index & similar neighbor index vector
        """The forward pass of the network
        
        Args:
            x_in (torch.Tensor): an input data tensor. 
                x_in.shape should be (batch, dataset._max_seq_length)
        Returns:
            the resulting tensor. tensor.shape should be (batch, num_classes)
        """
        
        center_vector = self.in_emb(index_vec)
        neighbor_vector = self.in_emb(neighbor_index_vec)
        
        euclidean_distance_1 = torch.dist(center_vector, neighbor_vector, p=2) #broadcasting...

        if(exist_non_neighbor):
            non_neighbor_vector1 = self.in_emb(non_neighbor1)
            non_neighbor_vector2 = self.in_emb(non_neighbor2)
            euclidean_distance_2 = torch.dist(center_vector, non_neighbor_vector1, p=2)
            euclidean_distance_3 = torch.dist(center_vector, non_neighbor_vector2, p=2)
            
        
        const_node = torch.tensor(0.1)
        criterion = nn.L1Loss()
        if(euclidean_distance_1 - radius_sum < 0):
            loss_1 = 10 * criterion(euclidean_distance_1, radius_sum)
        else:
            loss_1 = criterion(euclidean_distance_1, radius_sum)
        '''
        act = nn.ReLU()
        loss_1 = criterion(euclidean_distance_1, radius_sum)
        '''
        '''
        if(exist_non_neighbor):
            loss_2 = criterion(euclidean_distance_1, radius_sum)
            loss_3 = criterion(euclidean_distance_1, radius_sum)
            loss_2_log = -torch.log(loss_2 /radius_sum)
            loss_3_log = -torch.log(loss_3 /radius_sum)
            loss_2_act = act(loss_2_log)
            loss_3_act = act(loss_3_log)
            loss = loss_1 + loss_2_act + loss_3_act
            return loss
        '''
        return loss_1

    def get_embeddings(self):
        #get self.in_embed embeddings
        return self.in_emb.weight.detach().numpy()


class BoxNetwork(nn.Module):
    def __init__(self, center_size, entity_size, embeddings):
        """
        Args:
            center_size (int): size of the center vectors
            entity_size (int): number of entities(embed)

        the center vector matrix is considered as the parameter
        """
        super(BoxNetwork, self).__init__()

        embeddings = torch.from_numpy(embeddings).float()
        self.in_emb = nn.Embedding(num_embeddings=entity_size,
                                embedding_dim=center_size,
                                _weight=embeddings)
        self.in_emb.weight.requires_grad = True
        #from one-hot to center-vector

    def forward(self, index_vec, neighbor_index_vec, len_sum): #input: one-hot index & similar neighbor index vector
        """The forward pass of the network
        
        Args:
            x_in (torch.Tensor): an input data tensor. 
                x_in.shape should be (batch, dataset._max_seq_length)
        Returns:
            the resulting tensor. tensor.shape should be (batch, num_classes)
        """
        
        center_vector = self.in_emb(index_vec)
        neighbor_vector = self.in_emb(neighbor_index_vec)
        
        min_distance = math.inf
        for i in range(50):
            d = abs(center_vector[0][i] - neighbor_vector[0][i])
            if(d < min_distance):
                min_distance = d

        criterion = nn.L1Loss()

        if(min_distance < len_sum):
            loss = 100 * criterion(min_distance, len_sum)
        else:
            loss = criterion(min_distance, len_sum)
        return loss

    def get_embeddings(self):
        #get self.in_embed embeddings
        return self.in_emb.weight.detach().numpy()

def add_non_neighbor(one_item, balls_table, base_id, neighbor_dic, peers, exist_non_neighbor):
    if(exist_non_neighbor!=1):
        one_item.append([-1])
        one_item.append([-1])
        return -1
    non_neighbor = random.sample(peers, 1)[0]
    state = 0
    while(state == 0):
        if(non_neighbor not in neighbor_dic[base_id]):
            state = 1
            one_item.append([balls_table[non_neighbor]])
        else:
            non_neighbor = random.sample(peers, 1)[0]
    non_neighbor_2 = random.sample(peers, 1)[0]
    state = 0
    while(state == 0):
        if((non_neighbor_2 not in neighbor_dic[base_id]) and (non_neighbor_2 != non_neighbor)):
            state = 1
            one_item.append([balls_table[non_neighbor_2]])
        else:
            non_neighbor_2 = random.sample(peers, 1)[0]
    return 0
    


def get_network_inputs(neighbors, balls_table, peers, neighbor_dic, exist_non_neighbor):
    index_mat = []
    for neighbor in neighbors:
        one_item = []
        one_item.append([balls_table[neighbor[0]]])
        one_item.append([balls_table[neighbor[2]]])
        add_non_neighbor(one_item, balls_table, neighbor[0], neighbor_dic, peers, exist_non_neighbor)
        index_mat.append(np.array(one_item, float))
    return index_mat

def get_network_inputs_box(neighbors, balls_table, peers, neighbor_dic):
    index_mat = []
    for neighbor in neighbors:
        one_item = []
        one_item.append([balls_table[neighbor[0]]])
        one_item.append([balls_table[neighbor[2]]])
        index_mat.append(np.array(one_item, float))
    return index_mat


def get_layer_neighbor_dic(neighbors):
    neighbor_dic = {}
    for neighbor in neighbors:
        if neighbor[0] not in neighbor_dic:
            neighbor_dic[neighbor[0]] = [neighbor[2]]
        else:
            neighbor_dic[neighbor[0]].append(neighbor[2])
    return neighbor_dic

def determine_non_neighbor(neighbors, peers, neighbor_dic):
    for neighbor in neighbors:
        if(len(peers) - len(neighbor_dic[neighbor[0]]) - 1 < 2):
            return 0
    return 1


def adjust_neighbors(root, neighbors, balls_table, balls_matrix, sub_dic, parameters): #return the shift between "after training" and "before training"
    print("Start adjust children of ", funcs.get_label(root), " ...")
    
    balls = balls_matrix[:, :-1]
    peers = sub_dic[funcs.get_id(root)]

    neighbor_dic = get_layer_neighbor_dic(neighbors)
    
    exist_non_neighbor = determine_non_neighbor(neighbors, peers, neighbor_dic)

    if(parameters["neighbor_extension"] == True):
        #option for neighbor extension
        if(len(neighbors)<6):
            neighbors = neighbors * 2 
    

    entity_index_mat= get_network_inputs(neighbors, balls_table, peers, neighbor_dic, exist_non_neighbor) #the para matrix is for this layer
    model = Network(balls.shape[1], len(balls_matrix), balls)
    local_learning_rate = parameters["learning_rate"]
    optimizer = torch.optim.SGD(model.parameters(), lr=local_learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.975)

    old_loss_list_np = np.array([math.inf] * len(entity_index_mat))
    delta_loss_in_time_series = []
    loss_in_time_series = []
    epoch_list = []
    last_10_delta_loss = [math.inf] * 10
    learning_rate_count = 0

    for epoch in range(5000):
        loss_list = []
        random.shuffle(entity_index_mat)
        for index_pair in entity_index_mat:
            input0 = torch.from_numpy(index_pair[0]).long()
            input1 = torch.from_numpy(index_pair[1]).long()
            
            index0 = int(index_pair[0])
            index1 = int(index_pair[1])

            radius_sum = torch.tensor(balls_matrix[index0][-1]) + torch.tensor(balls_matrix[index1][-1])
            
            
            if(exist_non_neighbor):
                input2 = torch.from_numpy(index_pair[2]).long()
                input3 = torch.from_numpy(index_pair[3]).long()

                index2 = int(index_pair[2])
                index3 = int(index_pair[3])

                radius_sum_2 = torch.tensor(balls_matrix[index0][-1]) + torch.tensor(balls_matrix[index2][-1])
                radius_sum_3 = torch.tensor(balls_matrix[index0][-1]) + torch.tensor(balls_matrix[index3][-1])
            

            optimizer.zero_grad()
            
            if(exist_non_neighbor):
                loss = model.forward(input0, input1, input2, input3, radius_sum, radius_sum_2, radius_sum_3, exist_non_neighbor)
            else:
                loss = model.forward(input0, input1, -1, -1, radius_sum, -1, -1, exist_non_neighbor)
            

            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            #print("epoch: ", epoch, ", loss: ", loss.item())
    

        loss_in_time_series.append(np.array(loss_list).sum())
        delta = np.absolute(old_loss_list_np - np.array(loss_list)) #absolute...
        delta_sum = delta.sum() #sum loss of all neighbor pairs for one base ball
        old_loss_list_np = np.array(loss_list)

        last_10_delta_loss.pop(0)
        last_10_delta_loss.append(delta_sum)

        delta_loss_in_time_series.append(delta_sum)
        epoch_list.append(epoch + 1)
        last_10_delta_loss_avg = sum(last_10_delta_loss)/10

        learning_rate_count = learning_rate_count + 1

        scheduler.step()

        print("epoch: %d, last 10 average delta_loss: %.10f, current loss: %.10f" % (epoch + 1, last_10_delta_loss_avg, loss_in_time_series[-1]))
        if(last_10_delta_loss_avg < 0.001):#0.0002
            break
    
    plt.cla()
    plt.title('Delta Loss of ' + funcs.get_label(root) + " is: " + "%.6f" % (last_10_delta_loss_avg))
    plt.plot(epoch_list, delta_loss_in_time_series)
    plt.savefig('./pics/delta-loss-' + funcs.get_label(root) + '.png')

    if(len(entity_index_mat)!=0):
        average_loss = loss_in_time_series[-1]/len(entity_index_mat)
    else:
        average_loss = 0

    plt.cla()
    plt.title('Average Loss of ' + funcs.get_label(root) + " is " + "%.6f" % (average_loss))
    plt.plot(epoch_list, loss_in_time_series)
    plt.savefig('./pics/loss-' + funcs.get_label(root) + '.png')
    
    #plt.show()
    
    final_vecs = model.get_embeddings()
    delta_vecs = final_vecs - balls_matrix[:, :-1]
    balls_matrix[:, :-1] = final_vecs
    zeros = np.zeros([balls_matrix.shape[0],1])
    return np.concatenate((delta_vecs, zeros), axis = 1)
        

def training_one_family(root, layer_embedding, depth, balls_table, balls_matrix, word_to_idx, embeddings, root_area, children_neighbor, sub_dic, area_dic, coor_dic, parameters):
    #include children_neighbor as Cache.
    print("Constructing " + funcs.get_label(root) + "...")
    layer_vector_list = layer_embedding + [0] * (depth - len(layer_embedding)) #vector encoding the position info
    layer_vector = np.array(layer_vector_list) #convert from list to array
    if(len(layer_embedding) < depth):
        children = cons.get_children(root) #return = [{label: property}, {label2: peoperty2}]   property = {"label": Germany, "subdivision" = xxx}
        if(len(children) > 0):
            for i in range(len(children)):
                training_one_family(children[i], layer_embedding + [i + 1], depth, balls_table, balls_matrix, word_to_idx, embeddings, root_area, children_neighbor, sub_dic, area_dic, coor_dic, parameters)
            
            neighbors = []
            neighbors =  children_neighbor[funcs.get_id(root)]
            delta_vecs = adjust_neighbors(root, neighbors, balls_table, balls_matrix, sub_dic, parameters)
            for child in children:
                child_id = funcs.get_id(child)
                child_index = balls_table[child_id]
                cons.shift_sub_tree_of_root(child, delta_vecs[child_index], balls_table, balls_matrix)
            
            cons.create_parent_ball_of(root, children, layer_vector, balls_table, balls_matrix, word_to_idx, embeddings, depth, root_area, area_dic, parameters)
        else:
            cons.initialize_ball(root, layer_vector, balls_table, balls_matrix, word_to_idx, embeddings, root_area, area_dic, coor_dic, parameters)
    else:
        cons.initialize_ball(root, layer_vector, balls_table, balls_matrix, word_to_idx, embeddings, root_area, area_dic, coor_dic, parameters)


def count_neighbor_info(root, neighbor_count_dic, neighbor_dic, children_neighbor, sub_count_dic):
    root_label = funcs.get_label(root)
    root_id = funcs.get_id(root)
    print("Start counting " + root_label + " (" + root_id + ")")
    
    children = cons.get_children(root)
    if(len(children)>0):
        layer_neighbors = []
        new_neighbors = []
        sub_count_list = []
        children_id = [funcs.get_id(children[j]) for j in range(len(children))]
        for i in range(len(children)):
            sub_count_list.append(funcs.get_id(children[i]))
            count_neighbor_info(children[i], neighbor_count_dic, neighbor_dic, children_neighbor, sub_count_dic)
            new_neighbors = get_neighbor_list(funcs.get_label(children[i]), funcs.get_id(children[i]))
            neighbor_count_dic[funcs.get_id(children[i])] = [funcs.get_label(children[i]), len(new_neighbors)]
            layer_neighbors = layer_neighbors + new_neighbors

            neighbor_dic[funcs.get_id(children[i])] = [new_neighbors[k][2] for k in range(len(new_neighbors)) if new_neighbors[k][2] in children_id]
        layer_neighbors = cons.parent_filter(layer_neighbors)
        children_neighbor[root_id] = layer_neighbors.copy() #id to list
        sub_count_dic[root_id] = sub_count_list.copy()
    else:
        children_neighbor[root_id] = []
        sub_count_dic[root_id] = []

def get_area_info(root, area_dic):
    root_id = funcs.get_id(root)
    radius = cons.get_ball_radius(root_id)
    area_dic[root_id] = math.sqrt(float(radius))
    children = cons.get_children(root)
    for i in range(len(children)):
        get_area_info(children[i], area_dic)

def training_box_family(root, layer_count, depth, boxes_table, boxes_matrix, children_neighbor, sub_dic, parameters):
    if(layer_count < depth):
        children = cons.get_children(root) #return = [{label: property}, {label2: peoperty2}]   property = {"label": Germany, "subdivision" = xxx}
        if(len(children) > 0):
            neighbors = []
            for i in range(len(children)):
                training_box_family(children[i], layer_count + 1, depth, boxes_table, boxes_matrix, children_neighbor, sub_dic, parameters)
            neighbors =  children_neighbor[funcs.get_id(root)]
            
            delta_vecs = adjust_box_neighbors(root, neighbors, boxes_table, boxes_matrix, sub_dic, parameters)
            
            for child in children:
                child_id = funcs.get_id(child)
                child_index = balls_table[child_id]
                cons.shift_sub_tree_of_root(child, delta_vecs[child_index], boxes_table, boxes_matrix)


def adjust_box_neighbors(root, neighbors, boxes_table, boxes_matrix, sub_dic, parameters): #return the shift between "after training" and "before training"
    print("Start adjust children of ", funcs.get_label(root), " ...")
    
    boxes = boxes_matrix[:, :-1]
    peers = sub_dic[funcs.get_id(root)]

    neighbor_dic = get_layer_neighbor_dic(neighbors)

    entity_index_mat= get_network_inputs_box(neighbors, boxes_table, peers, neighbor_dic) #the para matrix is for this layer
    model = BoxNetwork(boxes.shape[1], len(boxes_matrix), boxes)
    local_learning_rate = parameters["box_lr"]
    b_optimizer = torch.optim.SGD(model.parameters(), lr=local_learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(b_optimizer, step_size=1, gamma=0.99949)
    
    old_loss_list_np = np.array([math.inf] * len(entity_index_mat))
    delta_loss_in_time_series = []
    loss_in_time_series = []
    epoch_list = []
    last_10_delta_loss = [math.inf] * 10

    for epoch in range(5000):
        loss_list = []
        random.shuffle(entity_index_mat)
        for index_pair in entity_index_mat:
            input0 = torch.from_numpy(index_pair[0]).long()
            input1 = torch.from_numpy(index_pair[1]).long()
            
            index0 = int(index_pair[0])
            index1 = int(index_pair[1])

            len_sum = torch.tensor(boxes_matrix[index0][-1]) + torch.tensor(boxes_matrix[index1][-1])
            

            b_optimizer.zero_grad()
            
            loss = model.forward(input0, input1, len_sum)
            
            loss.backward()
            b_optimizer.step()
            loss_list.append(loss.item())
            #print("epoch: ", epoch, ", loss: ", loss.item())

        loss_in_time_series.append(np.array(loss_list).sum())
        delta = np.absolute(old_loss_list_np - np.array(loss_list)) #absolute...
        delta_sum = delta.sum() #sum loss of all neighbor pairs for one base ball
        old_loss_list_np = np.array(loss_list)

        last_10_delta_loss.pop(0)
        last_10_delta_loss.append(delta_sum)

        delta_loss_in_time_series.append(delta_sum)
        epoch_list.append(epoch + 1)
        last_10_delta_loss_avg = sum(last_10_delta_loss)/10


        print("epoch: %d, average of last 10 delta_loss: %.10f, current average loss: %.10f" % (epoch + 1, last_10_delta_loss_avg, loss_in_time_series[-1]/len(entity_index_mat)))
        '''
        if(last_10_delta_loss_avg < 0.001):#0.0002
            break
        '''
        scheduler.step()
    
    plt.cla()
    plt.title('Box Delta Loss of ' + funcs.get_label(root) + " is: " + "%.6f" % (last_10_delta_loss_avg))
    plt.plot(epoch_list, delta_loss_in_time_series)
    plt.savefig('./pics/box-delta-loss-' + funcs.get_label(root) + '.png')

    if(len(entity_index_mat)!=0):
        average_loss = loss_in_time_series[-1]/len(entity_index_mat)
    else:
        average_loss = 0

    plt.cla()
    plt.title('Box Average Loss of ' + funcs.get_label(root) + " is " + "%.6f" % (average_loss))
    plt.plot(epoch_list, loss_in_time_series)
    plt.savefig('./pics/box-loss-' + funcs.get_label(root) + '.png')
    
    #plt.show()
    
    final_vecs = model.get_embeddings()
    delta_vecs = final_vecs - boxes_matrix[:, :-1]
    boxes_matrix[:, :-1] = final_vecs
    zeros = np.zeros([boxes_matrix.shape[0],1])
    return np.concatenate((delta_vecs, zeros), axis = 1)

def test_once(parent_id, base_ball_id, children_neighbor, balls_table, balls_matrix, sub_count_dic, id_to_label, threshold):
    criterion = nn.L1Loss()
    norm = nn.Tanh()
    loss_list = []
    id_list = []

    ball_set = children_neighbor[parent_id]
    peers_set = sub_count_dic[parent_id]

    #print("Queried ball is: " + id_to_label[base_ball_id])
    for i in range(len(peers_set)):
        #i = len(peers_set) - i - 1 #why inverse?
        if(peers_set[i] != base_ball_id):
            base_np = balls_matrix[balls_table[base_ball_id]]
            peer_np = balls_matrix[balls_table[peers_set[i]]]
            base_ball = torch.from_numpy(base_np[ :-1]).float()
            peer_ball = torch.from_numpy(peer_np[ :-1]).float()
            radius_sum = torch.tensor(float(base_np[-1] + peer_np[-1]))
            distance = torch.dist(base_ball, peer_ball, p=2)
            loss = criterion(distance, radius_sum)
            loss_list.append(loss.item())
            id_list.append(peers_set[i])

    loss_mean = np.mean(loss_list) #new
    loss_max = np.max(loss_list)

    #loss_list = loss_list - loss_mean
    #loss_list = loss_list - loss_max

    loss_tensor = torch.FloatTensor(loss_list)
    s = nn.Softmax(dim = 0)
    probability = s(torch.tensor(-1) * loss_tensor * torch.tensor(10))

    candidates = []
    for i in range(len(id_list)):
        if(probability.tolist()[i] >= threshold):
            candidates.append(id_list[i])

    p_list = []
    for i in range(len(id_list)):
        p_list.append([id_to_label[id_list[i]], probability.tolist()[i]]) #[[label, prob], [], ...]

    neighbors= []
    for i in range(len(ball_set)):
        if(ball_set[i][0]==base_ball_id):
            neighbors.append(id_to_label[ball_set[i][2]])

    def takeSecond(elem):
        return elem[1]

    p_list.sort(key=takeSecond, reverse=True)
    '''
    print("p_list is: ")
    for i in range(len(p_list)):
        print(p_list[i])
    print("neighbor is: ")
    print(neighbors)
    '''
    return candidates

def box_test_once(parent_id, base_box_id, children_neighbor, balls_table, balls_matrix, sub_count_dic, id_to_label, threshold):
    criterion = nn.L1Loss()
    norm = nn.Tanh()
    loss_list = []
    id_list = []

    ball_set = children_neighbor[parent_id]
    peers_set = sub_count_dic[parent_id]

    #print("Queried ball is: " + id_to_label[base_box_id])
    for i in range(len(peers_set)):
        #i = len(peers_set) - i - 1 #why inverse?
        if(peers_set[i] != base_box_id):
            base_np = balls_matrix[balls_table[base_box_id]]
            peer_np = balls_matrix[balls_table[peers_set[i]]]
            base_box = torch.from_numpy(base_np[ :-1]).float()
            peer_box = torch.from_numpy(peer_np[ :-1]).float()

            min_distance = math.inf
            for j in range(50):
                d = abs(base_box[j] - peer_box[j])
                if(d < min_distance):
                    min_distance = d

            len_sum = torch.tensor(float(base_np[-1] + peer_np[-1]))
            loss = criterion(min_distance, len_sum)
            loss_list.append(loss.item())
            id_list.append(peers_set[i])

    loss_mean = np.mean(loss_list) #new
    loss_max = np.max(loss_list)

    #loss_list = loss_list - loss_mean
    #loss_list = loss_list - loss_max

    loss_tensor = torch.FloatTensor(loss_list)
    s = nn.Softmax(dim = 0)
    probability = s(torch.tensor(-1) * loss_tensor)

    candidates = []
    for i in range(len(id_list)):
        if(probability.tolist()[i] >= threshold):#need to validate
            candidates.append(id_list[i])

    p_list = []
    for i in range(len(id_list)):
        p_list.append([id_to_label[id_list[i]], probability.tolist()[i]]) #[[label, prob], [], ...]

    neighbors= []
    for i in range(len(ball_set)):
        if(ball_set[i][0]==base_box_id):
            neighbors.append(id_to_label[ball_set[i][2]])

    def takeSecond(elem):
        return elem[1]

    p_list.sort(key=takeSecond, reverse=True)
    '''
    print("p_list is: ")
    for i in range(len(p_list)):
        print(p_list[i])
    print("neighbor is: ")
    print(neighbors)
    '''
    return candidates

def get_one_ball_loss(root_id, neighbor_dic, balls_table, balls_matrix):
    if(root_id == ROOT_ID):
        return 0
    criterion = nn.L1Loss()
    loss_list = []
    id_list = neighbor_dic[root_id]

    for i in range(len(id_list)):
        base_np = balls_matrix[balls_table[root_id]]
        neighbor_np = balls_matrix[balls_table[id_list[i]]]
        base_ball = torch.from_numpy(base_np[ :-1]).float()
        neighbor_ball = torch.from_numpy(neighbor_np[ :-1]).float()
        radius_sum = torch.tensor(float(base_np[-1] + neighbor_np[-1]))
        distance = torch.dist(base_ball, neighbor_ball, p=2)
        loss = criterion(distance, radius_sum)
        loss_list.append(loss.item())
    
    return np.array(loss_list).sum()

def get_one_box_loss(root_id, neighbor_dic, balls_table, balls_matrix):
    if(root_id == ROOT_ID):
        return 0
    criterion = nn.L1Loss()
    loss_list = []
    id_list = neighbor_dic[root_id]

    for i in range(len(id_list)):
        base_np = balls_matrix[balls_table[root_id]]
        neighbor_np = balls_matrix[balls_table[id_list[i]]]
        base_box = torch.from_numpy(base_np[ :-1]).float()
        neighbor_box = torch.from_numpy(neighbor_np[ :-1]).float()

        min_distance = math.inf
        for j in range(50):
            d = abs(base_box[j] - neighbor_box[j])
            if(d < min_distance):
                min_distance = d

        len_sum = torch.tensor(float(base_np[-1] + neighbor_np[-1]))
        loss = criterion(min_distance, len_sum)
        loss_list.append(loss.item())
    
    return np.array(loss_list).sum()

def get_all_loss(entity_list, neighbor_dic, balls_table, balls_matrix):
    loss_for_all_balls = {}
    for i in range(len(entity_list)):
        loss_for_all_balls[entity_list[i][0]] = get_one_ball_loss(entity_list[i][0], neighbor_dic, balls_table, balls_matrix)
    with open("./all_loss.json","w") as f:
        json.dump(loss_for_all_balls,f)

def get_all_box_loss(entity_list, neighbor_dic, balls_table, balls_matrix):
    loss_for_all_balls = {}
    for i in range(len(entity_list)):
        loss_for_all_balls[entity_list[i][0]] = get_one_box_loss(entity_list[i][0], neighbor_dic, balls_table, balls_matrix)
    with open("./all_loss.json","w") as f:
        json.dump(loss_for_all_balls,f)

def true_number(predict_result, real): #predict_result: list of candidates [id1, id2, ...], return true num
    true_count = 0
    for one_item in real:
        if(one_item in predict_result):
            true_count = true_count + 1
    return true_count

def recall_precision_rate(predict_dic, neighbor_dic): #input predict_dic: {id: [id1, id2, ...]}
    recall_all = 0
    precision_all = 0
    true_num = 0
    for key, value in predict_dic.items():
        true_num = true_num + true_number(value, neighbor_dic[key])
        recall_all = recall_all + len(neighbor_dic[key])
        precision_all = precision_all + len(value)
    return true_num / recall_all, true_num / precision_all



def prepare_dics_and_lists(parameters):
    # 1st step: create division tree, get entity list
    division_tree = funcs.get_tree_from_json(parameters)
    entity_list = cons.get_entity_list(division_tree) #[[Q,label], [Q, label], ...]
    funcs.find_duplicate(entity_list)

    # 2nd step: get corresponding embedding for each entity
    word_to_idx_ori, glove_embeddings_ori = load_glove_from_file(parameters["glove_filepath"]) #original embeddings
    word_to_idx, final_embedding, matching_dic = get_embeddings(parameters, entity_list) #mine to be compared

    # 3rd step: build dics and parameter matrix to be trained
    balls_matrix = np.zeros((len(entity_list), parameters["embedding_len"] + parameters["max_depth"] + parameters["coordinates_len"] + 1), float) #parameter matrix to be trained
    balls_table = {}
    id_to_label = {}
    label_to_id = {} #dublicate????
    for i in range(len(entity_list)):
        balls_table[entity_list[i][0]] = i
        id_to_label[entity_list[i][0]] = entity_list[i][1]
        label_to_id[entity_list[i][1]] = entity_list[i][0]
    
    

    return division_tree, entity_list, word_to_idx, final_embedding, matching_dic, balls_matrix, balls_table, id_to_label, label_to_id 

def get_file_related_var(parameters):
    # 4th step: initialize neighbordic(sub_dic), area_dic
    neighbor_dic = funcs.get_dic_from_json(parameters["neighbor_dic_path"])

    if(parameters["sub_dic_from_file"]==True):
        sub_dic = funcs.get_dic_from_json(parameters["sub_dic_path"])

    if(parameters["area_from_file"]==True):
        area_dic = funcs.get_dic_from_json(parameters["area_path"])
        root_area = area_dic[funcs.get_id(division_tree)]
        print("Load radius info from file.")
    else: 
        root_area = math.sqrt(float(cons.get_ball_radius(funcs.get_id(division_tree))))
    
    if(parameters["area_from_file"]==True and parameters["sub_dic_from_file"]==True):
        return neighbor_dic, sub_dic, area_dic, root_area #both true
    
def save_balls(balls_matrix):
    balls_df = pd.DataFrame(balls_matrix)
    balls_df.to_csv('balls_matrix.csv', index=False)

def store_area_dic(division_tree, parameters):
    area_dic = {}
    get_area_info(division_tree, area_dic)
    funcs.dic_to_json(area_dic, parameters["area_path"])

def store_node_information(division_tree, parameters):
    # generate dics
    neighbor_count_dic = {} #txt: count how many neighbors does a node have
    neighbor_dic = {} #neighbor of each node
    children_neighbor = {} #dic: count for the (child, neighbor) pair for each node
    sub_count_dic = {} #dic: count for id of children(sub-nodes) for each node
    count_neighbor_info(division_tree, neighbor_count_dic, neighbor_dic, children_neighbor, sub_count_dic)
    
    funcs.dic_to_json(sub_count_dic, parameters["sub_dic_path"])
    funcs.dic_to_json(neighbor_dic, parameters["neighbor_dic_path"])

    with open("neighbor_count.txt", "w", encoding='utf-8') as f:
        for key, value in neighbor_count_dic.items():
            f.write(key + "(" + value[0] + ")" + ": " + str(value[1]) + "\n")

    funcs.dic_to_json(children_neighbor, parameters["children_neighbor_path"])

def get_predict_dic(root, predict_dic, children_neighbor, balls_table, balls_matrix, sub_dic, id_to_label, threshold):
    root_id = funcs.get_id(root)
    #print("Predicting result with " +  root_id + "...")
    children = cons.get_children(root)
    for i in range(len(children)):
        child_id = funcs.get_id(children[i])
        candidates = test_once(root_id, child_id, children_neighbor, balls_table, balls_matrix, sub_dic, id_to_label, threshold)
        predict_dic[child_id] = candidates
        get_predict_dic(children[i], predict_dic, children_neighbor, balls_table, balls_matrix, sub_dic, id_to_label, threshold)

def get_box_predict_dic(root, predict_dic, children_neighbor, balls_table, balls_matrix, sub_dic, id_to_label, threshold):
    root_id = funcs.get_id(root)
    #print("Predicting result with " +  root_id + "...")
    children = cons.get_children(root)
    for i in range(len(children)):
        child_id = funcs.get_id(children[i])
        candidates = box_test_once(root_id, child_id, children_neighbor, balls_table, balls_matrix, sub_dic, id_to_label, threshold)
        predict_dic[child_id] = candidates
        get_box_predict_dic(children[i], predict_dic, children_neighbor, balls_table, balls_matrix, sub_dic, id_to_label, threshold)

def determine_pair_overlap(base_id, peer_id, balls_table, balls_matrix):
    base_np = balls_matrix[balls_table[base_id]]
    base_ball = torch.from_numpy(base_np[ :-1]).float()
    peer_np = balls_matrix[balls_table[peer_id]]
    peer_ball = torch.from_numpy(peer_np[ :-1]).float()
    radius_sum = torch.tensor(float(base_np[-1] + peer_np[-1]))
    distance = torch.dist(base_ball, peer_ball, p=2)
    return distance < radius_sum

def determine_box_pair_overlap(base_id, peer_id, balls_table, balls_matrix):
    base_np = balls_matrix[balls_table[base_id]]
    base_box = torch.from_numpy(base_np[ :-1]).float()
    peer_np = balls_matrix[balls_table[peer_id]]
    peer_box = torch.from_numpy(peer_np[ :-1]).float()
    len_sum = torch.tensor(float(base_np[-1] + peer_np[-1]))

    min_distance = math.inf
    for i in range(50):
        d = abs(base_box[i] - peer_box[i])
        if(d < min_distance):
            min_distance = d

    return min_distance < len_sum

def determine_overlap(root_id, result_dic, balls_table, balls_matrix, sub_dic, id_to_label): #determine whether the children overlap, count for overlapping num
    #print("Determining overlapping result with " +  id_to_label[root_id] + "...")
    children_list = sub_dic[root_id]
    for i in range(len(children_list)):
        determine_overlap(children_list[i], result_dic, balls_table, balls_matrix, sub_dic, id_to_label)
        for j in range(i + 1 , len(children_list)):
            result_dic["all"] = result_dic["all"] + 1
            if(determine_pair_overlap(children_list[i], children_list[j], balls_table, balls_matrix) == True):
                result_dic["overlap"] = result_dic["overlap"] + 1

def determine_box_overlap(root_id, result_dic, balls_table, balls_matrix, sub_dic, id_to_label): #determine whether the children overlap, count for overlapping num
    #print("Determining overlapping result with " +  id_to_label[root_id] + "...")
    children_list = sub_dic[root_id]
    for i in range(len(children_list)):
        determine_box_overlap(children_list[i], result_dic, balls_table, balls_matrix, sub_dic, id_to_label)
        for j in range(i + 1 , len(children_list)):
            result_dic["all"] = result_dic["all"] + 1
            if(determine_box_pair_overlap(children_list[i], children_list[j], balls_table, balls_matrix) == True):
                result_dic["overlap"] = result_dic["overlap"] + 1

def layer_generator(root_id, sub_dic, layer_list, depth):
    #print("Determining overlapping result with " +  id_to_label[root_id] + "...")
    if(len(layer_list) < depth):
        layer_list.append([])
    layer_list[depth-1].append(root_id)
    children_list = sub_dic[root_id]
    for i in range(len(children_list)):
        layer_generator(children_list[i], sub_dic, layer_list, depth + 1)

def find_parent(id, sub_dic):
    for key, value in sub_dic.items():
        if(id in value):
            return key

def get_parent_dic(sub_dic, entity_list):
    parent_dic = {}
    for one in entity_list:
        if(one[0] != ROOT_ID):
            parent_dic[one[0]] = find_parent(one[0], sub_dic)
    with open("./parent_dic.json","w") as f:
        json.dump(parent_dic,f)

def predict_layer(depth, layer_list, children_neighbor, balls_table, balls_matrix, sub_dic, id_to_label, parent_dic, threshold):
    layer_entities = layer_list[depth - 1]
    recall_all = 0
    precision_all = 0
    true_num = 0
    predict_dic = {}
    for i in range(len(layer_entities)):
        base_id = layer_entities[i]
        parent_id = parent_dic[base_id]
        peers = sub_dic[parent_id]
        for peer_id in peers:
            if(peer_id != base_id):
                candidates = test_once(parent_id, base_id, children_neighbor, balls_table, balls_matrix, sub_dic, id_to_label, threshold)
        predict_dic[base_id] = candidates
    for key, value in predict_dic.items():
        addition_true_num = true_number(value, neighbor_dic[key])
        true_num = true_num + addition_true_num
        if(len(neighbor_dic[key]) != 0):
            local_recall = addition_true_num / len(neighbor_dic[key])
            if(local_recall < 0.5):
                print("Low recall: " + key)
        else:
            print("No neighbor: " + key)
        local_precision = addition_true_num / len(value)
        if(local_precision < 0.5):
            print("Low precision: " + key)

        recall_all = recall_all + len(neighbor_dic[key])
        precision_all = precision_all + len(value)
    rr = true_num / recall_all
    pr = true_num / precision_all
    f1 = (2 * pr * rr) / (pr + rr)
    return true_num / recall_all, true_num / precision_all, f1

def predict_box_layer(depth, layer_list, children_neighbor, balls_table, balls_matrix, sub_dic, id_to_label, parent_dic, threshold):
    layer_entities = layer_list[depth - 1]
    recall_all = 0
    precision_all = 0
    true_num = 0
    predict_dic = {}
    for i in range(len(layer_entities)):
        base_id = layer_entities[i]
        parent_id = parent_dic[base_id]
        peers = sub_dic[parent_id]
        for peer_id in peers:
            if(peer_id != base_id):
                candidates = box_test_once(parent_id, base_id, children_neighbor, balls_table, balls_matrix, sub_dic, id_to_label, threshold)
        predict_dic[base_id] = candidates
    for key, value in predict_dic.items():
        addition_true_num = true_number(value, neighbor_dic[key])
        true_num = true_num + addition_true_num
        if(len(neighbor_dic[key]) != 0):
            local_recall = addition_true_num / len(neighbor_dic[key])
            if(local_recall < 0.5):
                
                print("$$Low recall: " + key)
        else:
            
            print("@@ No neighbor: " + key)

        if(len(value) != 0):
            local_precision = addition_true_num / len(value)
            if(local_precision < 0.5):
                
                print("&&Low precision: " + key)
        else:
            
            print("##No prediction: " + key)
        recall_all = recall_all + len(neighbor_dic[key])
        precision_all = precision_all + len(value)
    rr = true_num / recall_all
    pr = true_num / precision_all
    f1 = (2 * pr * rr) / (pr + rr)
    return true_num / recall_all, true_num / precision_all, f1

def get_overlapping_rate(balls_table, balls_matrix, sub_dic, id_to_label):
    result_dic = {}
    result_dic["all"] = 0
    result_dic["overlap"] = 0
    determine_overlap(ROOT_ID, result_dic, balls_table, balls_matrix, sub_dic, id_to_label)
    print("Overlapping rate is " + str(result_dic["overlap"]/result_dic["all"]))

def get_box_overlapping_rate(result_dic, balls_table, balls_matrix, sub_dic, id_to_label):
    result_dic = {}
    result_dic["all"] = 0
    result_dic["overlap"] = 0
    determine_box_overlap(ROOT_ID, result_dic, balls_table, balls_matrix, sub_dic, id_to_label)
    print("Overlapping rate is " + str(result_dic["overlap"]/result_dic["all"]))

def neighbor_prediction(division_tree, children_neighbor, balls_table, balls_matrix, sub_dic, id_to_label, parameters):
    predict_result_dic = {}
    predict_dic = {}
    get_predict_dic(division_tree, predict_dic, children_neighbor, balls_table, balls_matrix, sub_dic, id_to_label, parameters["threshold"])
    #get_box_predict_dic(division_tree, predict_dic, children_neighbor, balls_table, balls_matrix, sub_dic, id_to_label, th/100)
    rr, pr = recall_precision_rate(predict_dic, neighbor_dic) #input predict_dic: {id: [id1, id2, ...]}
    f1 = (2 * pr * rr) / (pr + rr)
    predict_result_dic[str(parameters["threshold"])] = [rr, pr, f1]
    for key, value in predict_result_dic.items():
        print("Threshold of " + key + ", the recall rate, precision rate and F1 score are " + str(value))

def box_neighbor_prediction(division_tree, children_neighbor, balls_table, balls_matrix, sub_dic, id_to_label, parameters):
    predict_result_dic = {}
    predict_dic = {}
    get_box_predict_dic(division_tree, predict_dic, children_neighbor, balls_table, balls_matrix, sub_dic, id_to_label, parameters["threshold"])
    rr, pr = recall_precision_rate(predict_dic, neighbor_dic) #input predict_dic: {id: [id1, id2, ...]}
    f1 = (2 * pr * rr) / (pr + rr)
    predict_result_dic[str(parameters["threshold"])] = [rr, pr, f1]
    for key, value in predict_result_dic.items():
        print("Threshold of " + key + ", the recall rate, precision rate and F1 score are " + str(value))

#################################################################################################################
#################################################################################################################

parameters = {
    "embedding_from_file": True,

    "get_new_cache": False,
    "sub_dic_from_file": True,
    "sub_dic_path": "./sub_dic.json",
    "area_from_file": True,
    "area_path": "./area_info.json",
    "neighbor_dic_path": "./neighbor_dic.json",
    "coor_from_file": True,
    "coor_path": "./coor_dic.json",

    "neighbor_from_file": True,
    "children_neighbor_path": "./children_neighbor.json",

    "final_embedding_path": "./embeddings-3.csv", #path where embeddings will be stored, if not from file
    "from_embedding_path": "./embeddings-3.csv",

    "glove_filepath": "./glove.6B.50d.txt",
    "word_to_idx_path": "./word_to_idx-3.json", #path where word_to_index dictionary will be stored
    "match_path": "./match-3.json", #path where match dictionary will be stored
    "tree_path": "./treeGER-3-new.json", #tree generated from the json

    "max_depth": 3, #make sure that it corresponds to "***Tree.json"
    "embedding_len": 50,
    "coordinates_len": 2,

    "learning_rate": 0.005,
    "box_lr": 0.005,

    "neighbor_extension": False,
    "threshold": 0.08
}

division_tree, entity_list, word_to_idx, final_embedding, matching_dic, balls_matrix, balls_table, id_to_label, label_to_id = prepare_dics_and_lists(parameters)


if(parameters["get_new_cache"]):
    store_area_dic(division_tree, parameters)
    store_node_information(division_tree, parameters)


neighbor_dic, sub_dic, area_dic, root_area = get_file_related_var(parameters)

#coor_dic = {}
#cons.get_coor_info(division_tree, coor_dic, parameters)
with open(parameters["coor_path"],'r') as load_f:
        coor_dic = json.load(load_f)


# 5th step: train the balls
initial_layer_embedding = [1]
'''
if (parameters["neighbor_from_file"]==True):
    children_neighbor = funcs.get_neighbor_from_json(parameters)
    print("Load neighbor info from file.")
    training_one_family(division_tree, initial_layer_embedding, parameters["max_depth"], balls_table, balls_matrix, word_to_idx, final_embedding, root_area, children_neighbor, sub_dic, area_dic, coor_dic, parameters)
'''
#training_box_family(division_tree, 1, parameters["max_depth"], balls_table, balls_matrix, children_neighbor, sub_dic, parameters)

#################################################################################################
# Test
'''
layer_list = []
layer_generator(ROOT_ID, sub_dic, layer_list, 1)
'''
parent_dic = funcs.get_parent_dic_from_json()
children_neighbor = funcs.get_neighbor_from_json(parameters)

balls_df = pd.read_csv("./results/balls_matrix_scaled.csv")
balls_matrix = balls_df.values

get_overlapping_rate(balls_table, balls_matrix, sub_dic, id_to_label)
neighbor_prediction(division_tree, children_neighbor, balls_table, balls_matrix, sub_dic, id_to_label, parameters)