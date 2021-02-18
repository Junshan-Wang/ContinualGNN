import os
import sys
import random
import numpy as np 
import pickle
import json
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict

import torch

class MemoryHandler(object):

    def __init__(self, args):
        # strategy: random / class
        super(MemoryHandler, self).__init__()
        
        self.memory_size = args.memory_size
        self.p = args.p
        self.strategy = args.memory_strategy
        self.clock = 0

        self.memory = list()

        if self.strategy == 'class':
            self.data_size = 0
            self.data_size_per_class = defaultdict(int)
            self.memory_size_per_class = defaultdict(int)
            self.memory_per_class = defaultdict(list)
            self.memory_per_class_log = defaultdict(list)
            self.alpha = args.alpha



    def update(self, nodes, x = None, y = None, adj_lists = None):
        if self.strategy == 'class':
            importance = self._compute_node_importance(nodes, y, adj_lists)
            self._update_class(nodes, y[nodes], importance)
        else:
            self._update_random(nodes)
        self.clock += 1
        

    def _update_random(self, nodes):
        for i, node in enumerate(nodes):
            if node in self.memory:
                continue
            elif len(self.memory) < self.memory_size:
                self.memory.append(node)
            else:
                if random.random() > self.p:
                    continue
                replace_idx = random.randint(0, self.memory_size - 1)
                self.memory[replace_idx] = node


    def _update_class(self, nodes, y, importance):
        # calculate number of samples per class in memory
        self.data_size += len(nodes)
        for i in y:
            self.data_size_per_class[i] += 1
        for i in self.data_size_per_class:
            self.memory_size_per_class[i] = int(self.data_size_per_class[i] / self.data_size * self.memory_size)
            while self.memory_size_per_class[i] < len(self.memory_per_class[i]):
                replace_idx = random.randint(0, len(self.memory_per_class[i]) - 1)
                del(self.memory_per_class[i][replace_idx])
                del(self.memory_per_class_log[i][replace_idx])

        # update memory
        for i, node in enumerate(nodes):
            if node in self.memory_per_class[y[i]]:
                continue
            elif self.memory_size_per_class[y[i]] > len(self.memory_per_class[y[i]]):
                self.memory_per_class[y[i]] += [node]
                self.memory_per_class_log[y[i]] += [(node, int(y[i]), self.clock, importance[i])]
            else:
                prob = self.p * self.memory_size_per_class[y[i]] / self.data_size_per_class[y[i]] * (1 + self.alpha * importance[i])
                if random.random() > prob:
                    continue
                replace_idx = random.randint(0, len(self.memory_per_class[y[i]]) - 1)
                self.memory_per_class[y[i]][replace_idx] = node 
                self.memory_per_class_log[y[i]][replace_idx] = (node, int(y[i]), self.clock, float(importance[i]))
        
        self.memory.clear()
        for i, m in self.memory_per_class.items():
            self.memory += m



    def _compute_node_importance(self, nodes, labels, adj_lists):
        importance = list()
        for node in nodes:
            node_label = labels[node]
            neighbors = np.array(list(adj_lists[node]))
            neighbor_labels = np.array([node_label] + [labels[neighbor] for neighbor in neighbors if labels[neighbor] != -1])
            importance += [np.sum(neighbor_labels != node_label) / neighbor_labels.shape[0]]
        importance = np.array(importance)
        importance = importance * 10 - 5
        importance = 1 / (1 + np.exp(-importance))
        return importance


def load(file_name, args):
    if os.path.exists('M'):
        open_file = open('M', 'rb')
        return pickle.load(open_file)
    else:
        # Do not return a null memory!! Cause error!
        return None
    

def save(A, file_name):
    open_file = open('M', 'wb')
    open_file.write(pickle.dumps(A))

