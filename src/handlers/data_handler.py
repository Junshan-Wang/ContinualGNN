import os
import sys
import numpy as np 
import logging
import random
from collections import defaultdict


class DataHandler(object):

    def __init__(self):
        super(DataHandler, self).__init__()

    def load(self, data_name):
        self.data_name = data_name

        # Load attributes
        attributes_file_name = os.path.join('../data', data_name, 'attributes')
        self.features = np.loadtxt(attributes_file_name)

        # Load labels
        labels_file_name = os.path.join('../data', data_name, 'labels')
        labels = np.loadtxt(labels_file_name, dtype = np.int64)
        
        # Load graph
        edges_file_name = os.path.join('../data', data_name, 'edges')
        self.adj_lists = defaultdict(set)
        with open(edges_file_name) as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                node1, node2 = int(info[0]), int(info[1])

                self.adj_lists[node1].add(node2)
                self.adj_lists[node2].add(node1)
        
        # Generate node and label list
        self.labels = np.zeros(len(self.adj_lists), dtype=np.int64)
        self.labels[labels[:, 0]] = labels[:, 1]

        logging.info('Number of total Nodes / Edges: ' + str(len(self.adj_lists)) \
            + ' / ' + str(sum([len(v) for v in self.adj_lists.values()]) / 2))

        # Input & Output size
        self.feature_size = self.features.shape[1]
        self.label_size = np.unique(self.labels).shape[0]

        # Generate train & valid data (nodes)
        train_file_name = os.path.join('../data', data_name, 'train_nodes')
        self.train_nodes = np.loadtxt(train_file_name, dtype = np.int64)
        self.train_size = self.train_nodes.shape[0]

        valid_file_name = os.path.join('../data', data_name, 'valid_nodes')
        self.valid_nodes = np.loadtxt(valid_file_name, dtype = np.int64)
        self.valid_size = self.valid_nodes.shape[0]

        self.data_size = self.train_size + self.valid_size

        logging.info('Data: ' + self.data_name + '; Data size: ' + str(self.data_size) 
            + '; Train size: ' + str(self.train_size) 
            + '; Valid size: ' + str(self.valid_size))

    

    def split(self, valid_ratio = 0.3):
        np.random.shuffle(self.nodes_list)
        self.valid_size = int(self.data_size * valid_ratio)
        self.train_size = self.data_size - int(self.data_size * valid_ratio)
        self.valid_nodes = self.nodes_list[ : self.valid_size]
        self.train_nodes = self.nodes_list[self.valid_size : ]

        logging.info('Data: ' + self.data_name + '; Data size: ' + str(self.data_size) \
            + '; Train size: ' + str(len(self.train_nodes)) \
            + '; Valid size: ' + str(len(self.valid_nodes)))
