import os, sys
import numpy as np 


def preprocess_edges(node_dict):
    open_file = open('cora.cites')
    lines = open_file.readlines()
    open_file.close()

    open_file = open('edges', 'w')
    for line in lines:
        items = line.strip().split('\t')
        node1, node2 = items[0], items[1]
        if node1 not in node_dict:
            continue
        if node2 not in node_dict:
            continue
        if node1 == node2:
            continue
        open_file.write(str(node_dict[node1]) + '\t' + str(node_dict[node2]) + '\n')
    open_file.close()

    return node_dict

def preprocess_attributes_labels():
    open_file = open('cora.content')
    lines = open_file.readlines()
    open_file.close()

    attributes = np.zeros((2708, 1433), dtype=np.int64)
    labels = np.zeros((2708), dtype=np.int64)
    node_dict = dict()
    label_map = dict()
    for line in lines:
        items = line.strip().split('\t')
        node = items[0]
        if node not in node_dict:
            node_dict[node] = len(node_dict)	
        attributes[node_dict[node], :] = list(map(float, items[1: -1]))[: 1433]
        if not items[-1] in label_map:
            label_map[items[-1]] = len(label_map)
        labels[node_dict[node]] = label_map[items[-1]]

    open_file1 = open('attributes', 'w')
    for i in range(2708):
        for j in range(1433):
            open_file1.write(str(attributes[i, j]) + '\t')
        open_file1.write('\n')
    open_file1.close()

    open_file2 = open('labels', 'w')
    for i in range(2708):
        open_file2.write(str(i) + '\t' + str(labels[i]) + '\n')
    open_file2.close()

    return node_dict


if __name__ == '__main__':
    
    node_dict = preprocess_attributes_labels()
    preprocess_edges(node_dict)

