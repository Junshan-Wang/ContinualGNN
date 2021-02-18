import os 
import sys
import logging
import numpy as np 
from collections import defaultdict, deque
import random
import time
import torch

from handlers.stream_data_handler import StreamDataHandler
from handlers.model_handler import ModelHandler
from models.graph_sage import GraphSAGE


def detect(data, t, args):
    if args.detect_strategy == 'simple':
        return detect_simple(data, t, args)
    elif args.detect_strategy == 'bfs':
        return detect_bfs(data, t, args)


def detect_simple(data, t, args):
    # Load model
    model = load_model(data, t, args)

    # Get embeddings of all nodes in adjacent graphs using previous model
    h_pre = get_h(model, data.train_cha_nodes_list + data.train_old_nodes_list, t - 1, args)
    h_cur = get_h(model, data.train_cha_nodes_list + data.train_old_nodes_list, t, args)

    # Calculate delta of embeddings
    delta_h = np.sum(np.abs(h_cur - h_pre), 1) 

    new_nodes_size = int(len(data.train_cha_nodes_list) * args.new_ratio)
    threshold = delta_h[np.argpartition(delta_h, - new_nodes_size)[- new_nodes_size]]
    new_nodes = np.array(data.train_cha_nodes_list + data.train_old_nodes_list)[np.where(delta_h > threshold)[0]]
    new_nodes = list(set(new_nodes))

    logging.info('New Data Size: ' + str(len(new_nodes)) + ' Among All Data Size: ' + str(delta_h.shape[0]))
    return new_nodes

def detect_bfs(data, t, args):
    # Load model
    model = load_model(data, t, args)
    
    # Get embeddings of changed nodes in adjacent graphs using previous model
    h_pre = get_h(model, data.train_cha_nodes_list, t - 1, args)
    h_cur = get_h(model, data.train_cha_nodes_list, t, args)

    # Calculate delta of embeddings
    delta_h = np.sum(np.abs(h_cur - h_pre), 1)

    # Calculate influenced nodes
    nodelist, f_matrix = bfs_plus(data, data.train_cha_nodes_list, 2)
    f_matrix = np.dot(f_matrix, delta_h)

    new_nodes_size = int(f_matrix.shape[0] * args.new_ratio)
    threshold = f_matrix[np.argpartition(f_matrix, - new_nodes_size)[- new_nodes_size]]
    new_nodes = np.array(nodelist)[np.where(f_matrix > threshold)[0]]
    new_nodes = list(set(new_nodes).intersection(set(data.train_cha_nodes_list + data.train_old_nodes_list)))

    logging.info('New Data Size: ' + str(len(new_nodes)) + ' Among Neighborhood Size: ' + str(len(nodelist)))
    return new_nodes



def load_model(data, t, args):
    # Load model dict
    model_handler = ModelHandler(os.path.join(args.save_path, str(t - 1)))
    if t < 1 or model_handler.not_exist():
        logging.warn('Load model fail, do not detect new pattern!')
        return None
    
    # Load model
    layers = [data.feature_size] + [args.embed_size] * args.num_layers + [data.label_size]
    model = GraphSAGE(layers, data.features, data.adj_lists, args).to(args.device)
    model.load_state_dict(model_handler.load('graph_sage.pkl'))
    return model



def get_h(model, train_cha_nodes_list, t, args):
    # Load data for AdjLists
    data = AdjListsHandler()
    data.load(args.data, t)
    model.sampler.adj_lists = data.adj_lists
    
    # Get embeddings
    h = model.forward(train_cha_nodes_list).data.cpu().numpy()
    return h



class AdjListsHandler(StreamDataHandler):
    def load(self, data_name, t):
        self.data_name = data_name
        self.t = t

        # Load graph
        stream_edges_dir_name = os.path.join('../data', data_name, 'stream_edges')
        self.adj_lists = defaultdict(set)
        
        begin_time = max(0, t - 9)
        end_time = t
        for tt in range(0, len(os.listdir(os.path.join('../data', data_name, 'stream_edges')))):
            edges_file_name = os.path.join(stream_edges_dir_name, str(tt))
            with open(edges_file_name) as fp:
                for i, line in enumerate(fp):
                    info = line.strip().split()
                    node1, node2 = int(info[0]), int(info[1])
                    if tt <= end_time and tt >= begin_time:
                        self.adj_lists[node1].add(node2)
                        self.adj_lists[node2].add(node1)
        

def bfs_plus(data, nodes, hop = 2):
    adj_lists = data.adj_lists
    node2f = defaultdict(dict)
    node2idx = dict()
    center2idx = dict()

    for center in nodes:
        # Search from each changed nodes
        vis_nodes = dict()
        vis_nodes[center] = 1
        f = list()
        f.append(dict())
        f[0][center] = 1
        for i in range(hop):
            f.append(dict())
            new_nodes = dict()
            for node in vis_nodes:
                for neigh in adj_lists[node]:
                    if neigh not in vis_nodes and neigh not in new_nodes:
                        new_nodes[neigh] = 1
            vis_nodes.update(new_nodes)
            for node in vis_nodes:
                fun = 0.0
                if node in f[i]:
                    fun += f[i][node]
                for neigh in adj_lists[node]:
                    if neigh in f[i]:
                        fun += f[i][neigh]
                f[i + 1][node] = fun / (len(adj_lists[node]) + 1)
    
        for node in vis_nodes:
            if node not in node2idx:
                node2idx[node] = len(node2idx)
            node2f[node][center] = f[hop][node]
        if center not in center2idx:
            center2idx[center] = len(center2idx)
    
    f_matrix = np.zeros((len(node2idx), len(nodes)), dtype=np.float32)
    for node in node2f:
        for center in node2f[node]:
            f_matrix[node2idx[node]][center2idx[center]] = node2f[node][center]
    
    return list(node2idx.keys()), f_matrix

