import os
import sys
import random
import numpy as np 
from collections import defaultdict

def generate_dynamic_edges_data_random(stream_size = 200):
    ori_file_name = os.path.join('edges')
    edges = np.loadtxt(ori_file_name, dtype = np.int64)
    data_size = edges.shape[0]

    dir_name = 'stream_edges_' + str(stream_size)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    np.random.shuffle(edges)
    for i in range(data_size // stream_size + 1):
        stream_edges = edges[i * stream_size : (i + 1) * stream_size]
        new_file = open(os.path.join(dir_name, str(i)), 'w')
        for edge in stream_edges:
            new_file.write(str(edge[0]) + '\t' + str(edge[1]) + '\n')
        new_file.close()


def generate_test_edges_data():
    ori_file_name = os.path.join('edges')
    edges = np.loadtxt(ori_file_name, dtype = np.int64)
    data_size = edges.shape[0]

    dir_name = 'stream_edges_' + str('test')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    stream_idxs = [0, 5400, 100000]
    np.random.shuffle(edges)
    for i in range(len(stream_idxs) - 1):
        stream_edges = edges[stream_idxs[i] : stream_idxs[i + 1]]
        new_file = open(os.path.join(dir_name, str(i)), 'w')
        for edge in stream_edges:
            new_file.write(str(edge[0]) + '\t' + str(edge[1]) + '\n')
        new_file.close()


def generate_dynamic_edges_data_label():
    ori_file_name = os.path.join('edges')
    edges = np.loadtxt(ori_file_name, dtype = np.int64)
    ori_file_name = os.path.join('labels')
    labels = np.loadtxt(ori_file_name, dtype = np.int64)[:, 1]
    data_size = edges.shape[0]
    label_size = np.unique(labels).shape[0]

    dir_name = 'stream_edges_label'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    stream_edges = [[], [], [], [], [], [], []]
    stream_labels = np.zeros((label_size, label_size))

    # p1, p2 = 0.48, 0.96
    # for edge in edges:
    #     r = random.random()
    #     if r < p1:
    #         stream_edges[labels[edge[0]]].append(edge)
    #         stream_labels[labels[edge[0]], labels[edge[0]]] += 1
    #         stream_labels[labels[edge[0]], labels[edge[1]]] += 1
    #     elif r < p2:
    #         stream_edges[labels[edge[1]]].append(edge)
    #         stream_labels[labels[edge[1]], labels[edge[0]]] += 1
    #         stream_labels[labels[edge[1]], labels[edge[1]]] += 1
    #     else:
    #         l = random.randint(0, label_size - 1)
    #         stream_edges[l].append(edge)
    #         stream_labels[l, labels[edge[0]]] += 1
    #         stream_labels[l, labels[edge[1]]] += 1


    # for i, label in enumerate(range(label_size)):
    #     new_file = open(os.path.join(dir_name, str(i)), 'w')
    #     print(label, len(stream_edges), stream_labels[label])
    #     for edge in stream_edges[label]:
    #         new_file.write(str(edge[0]) + '\t' + str(edge[1]) + '\n')
    #     new_file.close()

    for i, label in enumerate(range(label_size)): # [0, 5, 2, 3, 1, 6, 4]
        stream_label_nodes1 = defaultdict(set)
        stream_label_nodes2 = defaultdict(set)
        new_file1 = open(os.path.join(dir_name, str(2 * i)), 'w')
        new_file2 = open(os.path.join(dir_name, str(2 * i + 1)), 'w')
        for edge in edges:
            if labels[edge[0]] == label or labels[edge[1]] == label:
                if edge[0] in stream_label_nodes1[labels[edge[0]]] or edge[1] in stream_label_nodes1[labels[edge[1]]]:
                    new_file1.write(str(edge[0]) + '\t' + str(edge[1]) + '\n')
                    stream_label_nodes1[labels[edge[0]]].add(edge[0])
                    stream_label_nodes1[labels[edge[1]]].add(edge[1])
                elif edge[0] in stream_label_nodes2[labels[edge[0]]] or edge[1] in stream_label_nodes2[labels[edge[1]]]:
                    new_file2.write(str(edge[0]) + '\t' + str(edge[1]) + '\n')
                    stream_label_nodes2[labels[edge[0]]].add(edge[0])
                    stream_label_nodes2[labels[edge[1]]].add(edge[1])
                elif random.random() > 0.6:
                    new_file1.write(str(edge[0]) + '\t' + str(edge[1]) + '\n')
                    stream_label_nodes1[labels[edge[0]]].add(edge[0])
                    stream_label_nodes1[labels[edge[1]]].add(edge[1])
                else:
                    new_file2.write(str(edge[0]) + '\t' + str(edge[1]) + '\n')
                    stream_label_nodes2[labels[edge[0]]].add(edge[0])
                    stream_label_nodes2[labels[edge[1]]].add(edge[1])
        new_file1.close()
        new_file2.close()
        print(label, [(l, len(stream_label_nodes1[l])) for l in range(label_size)], sum([len(stream_label_nodes1[l]) for l in range(label_size)]))
        print(label, [(l, len(stream_label_nodes2[l])) for l in range(label_size)], sum([len(stream_label_nodes2[l]) for l in range(label_size)]))


def generate_stream_edges_data():
    ori_file_name = os.path.join('edges')
    edges = np.loadtxt(ori_file_name, dtype = np.int64)
    ori_file_name = os.path.join('labels')
    labels = np.loadtxt(ori_file_name, dtype = np.int64)[:, 1]
    data_size = edges.shape[0]
    label_size = np.unique(labels).shape[0]

    data_size_per_time = 50

    dir_name = 'stream_edges_' + str(data_size_per_time)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    label2edges = defaultdict(list)
    for edge in edges:
        if random.random() > 0.5:
            label2edges[labels[edge[0]]].append([edge[0], edge[1]])
        else:
            label2edges[labels[edge[1]]].append([edge[0], edge[1]])
    
    all_edges = []
    for label in label2edges:
        all_edges += label2edges[label]
    print(len(all_edges))

    all_edges = set()
    cnt = 0
    for i, label in enumerate(range(label_size)): 
        es = label2edges[label]
        file_num = int(len(es) / data_size_per_time)
        file_es = defaultdict(list)
        for e in es:
            # print(file_num)
            idx = random.randint(0, file_num - 1)
            if i == 0:
                file_es[max(0, idx - 9)].append(e)
            else:
                file_es[idx].append(e)
        
        l = 0
        if i == 0:
            file_num -= 9
        for f in range(file_num):
            np.savetxt(os.path.join(dir_name, str(cnt + f)), np.array(file_es[f]), fmt='%d\t%d')
            # print(len(file_es[f]))
            l += len(file_es[f])
            for e in file_es[f]:
                all_edges.add((e[0], e[1]))

        print(len(es), l, file_num)

        cnt += file_num

    print(len(all_edges))


def generate_labeled_nodes():
    ori_file_name = os.path.join('edges')
    edges = np.loadtxt(ori_file_name, dtype = np.int64)
    data_size = edges.shape[0]

    nodes = set()
    for edge in edges:
        nodes.add(edge[0])
        nodes.add(edge[1])
    
    train_file = open('train_nodes', 'w')
    valid_file = open('valid_nodes', 'w')
    for node in nodes:
        if random.random() > 0.3:
            train_file.write(str(node) + '\n')
        else:
            valid_file.write(str(node) + '\n')
    train_file.close()
    valid_file.close()



random.seed(1)
np.random.seed(1)    
# generate_test_edges_data()
# generate_dynamic_edges_data_random(200)
# generate_dynamic_edges_data_label()
# generate_labeled_nodes()
generate_stream_edges_data()
