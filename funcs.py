import pandas as pd
import json
import matplotlib.pyplot as plt
import math

def get_id(root):
    for label, properties in root.items():
        return properties["label"]

def get_subdivision(root):
    for label, properties in root.items():
        return properties["subdivision"]

def get_label(root):
    for label, properties in root.items():
        return label

def embedding_to_csv(embedding, path):
    df = pd.DataFrame(embedding)
    df.to_csv(path, sep=',', index = False, encoding='utf-8')

def word_dic_to_json(dic, path):
    with open(path,"w") as f:
        json.dump(dic,f)

def match_dic_to_json(matching_dic, path):
    with open(path,"w") as f:
        json.dump(matching_dic,f)

def dic_to_json(dic, path):
    with open(path,"w") as f:
        json.dump(dic,f)

def get_tree_from_json(parameters):
    with open(parameters["tree_path"],'r') as load_f:
        root = json.load(load_f)
    return root

def get_neighbor_from_json(parameters):
    with open(parameters["children_neighbor_path"],'r') as load_f:
        dic = json.load(load_f)
    return dic

def get_parent_dic_from_json():
    with open("./parent_dic.json",'r') as load_f:
        dic = json.load(load_f)
    return dic

def get_dic_from_json(path):
    with open(path,'r') as load_f:
        dic = json.load(load_f)
    return dic

def get_proper_embed_index(label, word_to_index_dic):
    #print(label)
    label = label.lower()
    index_list = []
    word_list = []
    label_parts = label.split(" ")
    for part in label_parts:
        find = 0
        for word, index in word_to_index_dic.items():
            if(word.lower().find(part) >= 0):
                index_list.append(index)
                word_list.append(word)
                find = 1
                break
        if(find == 0):
            index_list.append(-1)
            word_list.append("NULL")
                
    
    return [index_list, word_list]

def print_matching_rate(matching_dic):
    length = len(matching_dic)
    print("Len = " + str(length))
    count = 0

    for key, value in matching_dic.items():
        for i, one_word in enumerate(value):
            if(one_word != "NULL"):
                count = count + 1
                break

    print("There are " + str(count) + " matching items.")
    print("Matching rate = " + str(count/length))

def find_duplicate(entity_list):
    duplicate_list = []
    for entity_index, one_entity in enumerate(entity_list):
        duplicate_list.append(one_entity[1])
    duplicate_dic = {}
    for name in duplicate_list:
        if(duplicate_list.count(name))>1:
            duplicate_dic[name] = duplicate_list.count(name)
    print("Duplicate_dic: " + str(duplicate_dic))

def get_word_embedding(label, word_to_idx, embeddings):
    return embeddings[word_to_idx[label]]


def get_conversion_rate(dimension):
    r = 1
    n = float(dimension)
    volume = (math.pi**(n/2)/math.gamma(n/2 + 1))*(r**n)
    leng = math.pow(volume, float(1)/n)
    return math.sqrt(n)*leng/r
