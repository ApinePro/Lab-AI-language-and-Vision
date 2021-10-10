from numpy.core.defchararray import center
from numpy.lib.function_base import average
from numpy.lib.shape_base import apply_along_axis
import pandas as pd
import numpy as np
from torch.nn.functional import cross_entropy, embedding

from typing import Awaitable
from qwikidata.sparql import (get_subclasses_of_item,
                              return_sparql_query_results)

import json
import funcs

import math

BREMEN = '317.88'

def get_entity_list(root):
    entity_list = []
    temp_list = []
    for sub_label, sub_properties in root.items():
        temp_list = [sub_properties["label"], sub_label]
        entity_list.append(temp_list)
        entity_list = entity_list + get_entity_list(sub_properties["subdivision"])
    return entity_list

def get_children(root): #return = [{label: property}, {label2: peoperty2}]   property = {"label": Germany, "subdivision" = xxx}
    final_list = []
    if(len(root) > 0):
        subdivision_dic = funcs.get_subdivision(root)
        for sub_label, sub_properties in subdivision_dic.items():
            temp_dic = {}
            temp_dic[sub_label] = sub_properties
            final_list.append(temp_dic)
    return final_list

def get_coordinates_query(queried_entity):
  part1 = """
  SELECT DISTINCT ?coor
  WHERE
  {
  """
  part2 = "<http://www.wikidata.org/entity/" + queried_entity + "> " + "wdt:P625 ?coor ."
  part3 = """
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
  }
  """
  return part1 + part2 + part3

def get_coordinates(entity_id):  #return: float
    sparql_query = get_coordinates_query(entity_id)
    error = False
    while(error == False):
        try:
            res = return_sparql_query_results(sparql_query)
        except BaseException:
            pass
        else:
            if(len(res["results"]["bindings"]) != 0):
                coor = res["results"]["bindings"][0]["coor"]["value"]
                error = True 
    coor = [float(c) for c in coor[6:-1].split(" ")] 
    return coor

def get_area_query(queried_entity):
  part1 = """
  SELECT DISTINCT ?area
  WHERE
  {
  """
  part2 = "<http://www.wikidata.org/entity/" + queried_entity + "> " + "wdt:P2046 ?area ."
  part3 = """
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
  }
  """
  return part1 + part2 + part3

def get_ball_radius(entity_id):  #return: float
    sparql_query = get_area_query(entity_id)
    error = False
    count = 0
    while(error == False and count<10):
        try:
            res = return_sparql_query_results(sparql_query)
        except BaseException:
            pass
        else:
            count = count + 1
            if(len(res["results"]["bindings"]) != 0):
                area = res["results"]["bindings"][0]["area"]["value"]
                error = True  
    if(count>=10):
        return BREMEN
    else:
        return area #but need to improve

def get_coor_info(root, coor_dic, parameters):
    root_id = funcs.get_id(root)
    print("Getting coor info of " + root_id)
    coor = get_coordinates(root_id)
    coor_dic[root_id] = coor
    children = get_children(root)
    for i in range(len(children)):
        get_coor_info(children[i], coor_dic, parameters)
    with open(parameters["coor_path"],"w") as f:
        json.dump(coor_dic,f)


def initialize_ball(root, layer_vector, balls_table, balls_matrix, word_to_idx, embeddings, root_area, area_dic, coor_dic, parameters):
    id = funcs.get_id(root)
    embedding_vector = embeddings[word_to_idx[funcs.get_label(root)]]
    if(parameters["coor_from_file"]):
        coor = coor_dic[id]
    else:
        coor = get_coordinates(id)
    coordinates_vector = np.array(coor, float)
    if(parameters["area_from_file"]):
        radius = np.array([area_dic[id]/root_area], float)
    else:
        radius = np.array([math.sqrt(float(get_ball_radius(id)))/root_area], float) #get a square root of the area as radius
    final_vector = np.concatenate((embedding_vector, layer_vector, coordinates_vector, radius))
    balls_matrix[balls_table[id]] = final_vector

def find_far_len(center_vec, children, balls_table, balls_matrix):
    len_of_vec = len(center_vec)
    length_list = [] 
    for child in children:
        child_vec = balls_matrix[balls_table[funcs.get_id(child)]]
        length_list.append(np.linalg.norm(center_vec[:len_of_vec - 1] - child_vec[:len_of_vec - 1]) + child_vec[len_of_vec - 1])
    return max(length_list)

def create_parent_ball_of(root, children, layer_vector, balls_table, balls_matrix, word_to_idx, embeddings, depth, root_area, area_dic, parameters):
    label = funcs.get_label(root)
    id = funcs.get_id(root)
    embedding_vector = funcs.get_word_embedding(label, word_to_idx, embeddings)
    #coordinates_vector = np.array(get_coordinates(id), float)
    #len_of_vec = parameters["embedding_len"] + parameters["extension_len"] + depth - 1
    #len_of_vec = parameters["embedding_len"] + depth
    #extension_position = len_of_vec - parameters["extension_len"]
    #average_extension_vector = np.zeros(parameters["extension_len"])
    layer_vector_avg = np.zeros(depth + 2)
    for child in children:
        child_layer_vec = balls_matrix[balls_table[funcs.get_id(child)]][parameters["embedding_len"] : parameters["embedding_len"] + depth + 2]
        layer_vector_avg = layer_vector_avg + child_layer_vec
    layer_vector_avg = layer_vector_avg / len(children)

    
    center_vector = np.concatenate((embedding_vector, layer_vector_avg))
    radius = find_far_len(np.concatenate((center_vector, np.array([0.0]))), children, balls_table, balls_matrix)
    if(parameters["area_from_file"]):
        radius_q = area_dic[id] / root_area
    else:
        radius_q = math.sqrt(float(get_ball_radius(id))) / root_area
    if(radius<radius_q):
        radius = radius_q
    radius_vector = np.array([radius])
    center_vector = np.concatenate((center_vector, radius_vector))
    balls_matrix[balls_table[id]] = center_vector

def shift_one_sub_tree(root, delta_vec, balls_table, balls_matrix):
    root_id = funcs.get_id(root)
    root_index = balls_table[root_id]
    balls_matrix[root_index] = balls_matrix[root_index] + delta_vec
    children = get_children(root)
    if(len(children) > 0):
        for child in children:
            shift_one_sub_tree(child, delta_vec, balls_table, balls_matrix)

def shift_sub_tree_of_root(root, delta_vec, balls_table, balls_matrix):
    children = get_children(root)
    if(len(children) > 0):
        for child in children:
            shift_one_sub_tree(child, delta_vec, balls_table, balls_matrix)


def parent_filter(neighbors):
    index_set = set([one_relation[0] for one_relation in neighbors])
    i = 0
    while(i<len(neighbors)):
        if(neighbors[i][2] not in index_set):
            neighbors.pop(i)
        else:
            i = i + 1
    return neighbors