import sys
import os
import torch
import random
import logging
import time
import math
import numpy as np 
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.autograd import Variable

import utils
from models.graph_sage import GraphSAGE
from models.ewc import EWC
from handlers.stream_data_handler import StreamDataHandler
from handlers.model_handler import ModelHandler
from extensions import detection
from extensions import memory_handler


def train(data, model, args):
    # Model training
    times = []
    for epoch in range(args.num_epochs):
        losses = 0
        start_time = time.time()

        nodes = data.train_nodes
        np.random.shuffle(nodes)
        for batch in range(len(nodes) // args.batch_size):
            batch_nodes = nodes[batch * args.batch_size : (batch + 1) * args.batch_size]
            batch_labels = torch.LongTensor(data.labels[np.array(batch_nodes)]).to(args.device)

            model.optimizer.zero_grad()
            loss = model.loss(batch_nodes, batch_labels)
            loss.backward()
            model.optimizer.step()

            loss_val = loss.data.item()
            losses += loss_val * len(batch_nodes)
            if (np.isnan(loss_val)):
                logging.error('Loss Val is NaN !!!')
                sys.exit()
        
        if epoch % 10 == 0:
            logging.debug('--------- Epoch: ' + str(epoch) + ' ' + str(np.round(losses / data.train_size, 10)))
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.round(np.mean(times), 6)
    logging.info("Average epochs time: " + str(avg_time))
    return avg_time


def run(args, t):
    # Data loader
    data = StreamDataHandler()
    data.load(args.data, t)
    logging.info('Number of ALL Nodes / Edges: ' + str(len(data.adj_lists)) \
    + ' / ' + str(sum([len(v) for v in data.adj_lists.values()]) / 2))
    logging.info('Data: ' + data.data_name + '; Data size: ' + str(data.data_size) \
    + '; Train size: ' + str(len(data.train_nodes)) \
    + '; Valid size: ' + str(len(data.valid_nodes)))


    if args.new_ratio > 0.0 and t > 0:
        data.train_new_nodes_list = detection.detect(data, t, args)
        data.train_nodes = list(set(data.train_nodes + data.train_new_nodes_list)) if len(data.train_new_nodes_list) > 0 else data.train_nodes
    else:
        detect_time = 0

    # Sage layers
    layers = [data.feature_size] + [args.embed_size] * args.num_layers + [data.label_size]
    
    # GraphSAGE definition & initialization
    sage = GraphSAGE(layers, data.features, data.adj_lists, args)

    if t > 0:
        model_handler_pre = ModelHandler(os.path.join(args.save_path, str(t - 1)))
        if not model_handler_pre.not_exist():
            sage.load_state_dict(model_handler_pre.load('graph_sage.pkl'))
        
    if t > 0:
        ewc_model = EWC(sage, args.ewc_lambda, args.ewc_type).to(args.device)

        # whether use memory to store important nodes
        if args.memory_size > 0:
            memory_h = memory_handler.load('M', args)
            important_nodes_list = memory_h.memory
            data.train_nodes = list(set(data.train_nodes + important_nodes_list))
            logging.info('Important Data Size: ' + str(len(important_nodes_list)) + ' / ' + str(len(data.train_nodes)))
        else:
            important_nodes_list = data.train_old_nodes_list
            
        # calculate weight importance
        ewc_model.register_ewc_params(important_nodes_list, torch.LongTensor(data.labels[important_nodes_list]).to(args.device))
    else:
        if args.memory_size > 0:
            memory_h = memory_handler.MemoryHandler(args)
        ewc_model = sage.to(args.device)
            
    # Train
    ewc_model.optimizer = torch.optim.SGD(sage.parameters(), lr = args.learning_rate)
    avg_time = train(data, ewc_model, args)

    # Model save
    model_handler_cur = ModelHandler(os.path.join(args.save_path, str(t)))
    model_handler_cur.save(sage.state_dict(), 'graph_sage.pkl')

    # Memory save
    if args.memory_size > 0:
        train_output = sage.forward(data.train_nodes).data.cpu().numpy()
        memory_h.update(data.train_nodes, x=train_output, y=data.labels, adj_lists=data.adj_lists)
        memory_handler.save(memory_h, 'M')

    return avg_time


def evaluate(args, t):
    # Data loader
    data = StreamDataHandler()
    data.load(args.data, t)

    # Sage layers
    layers = [data.feature_size] + [args.embed_size] * args.num_layers + [data.label_size]
    
    # GraphSAGE definition
    sage = GraphSAGE(layers, data.features, data.adj_lists, args).to(args.device)

    # GraphSAGE load
    model_handler_cur = ModelHandler(os.path.join(args.save_path, str(t)))
    sage.load_state_dict(model_handler_cur.load('graph_sage.pkl'))

    valid_nodes = data.valid_nodes

    if len(valid_nodes) == 0:
        return 0, 0

    valid_output = sage.forward(valid_nodes).data.cpu().numpy().argmax(axis=1)
    f1, acc = utils.node_classification(data.labels[valid_nodes], valid_output, '')

    return f1, acc


if __name__ == "__main__":
    args = utils.parse_argument()
    if args.eval:
        logging.basicConfig(level = logging.WARNING, format = '%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s') # filename = 'log', 
    utils.print_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.device = utils.check_device(args.cuda)

    # f1, acc, time, detect_time
    print_ans = ['', '', '', '']
    avg_ans = [0.0, 0.0, 0.0, 0.0]

    t_num = len(os.listdir(os.path.join('../data', args.data, 'stream_edges')))
    args.save_path = os.path.join('../res', args.data)
    
    for t in range(0, t_num):
        logging.info('-------- Time ' + str(t) + ' --------')
        if args.eval == False:
            b = run(args, t)
        else:
            b = 0

        a = evaluate(args, t)
        print_ans[0] += str(a[0]) + '\t'
        print_ans[1] += str(a[1]) + '\t'
        print_ans[2] += str(b) + '\t'
        avg_ans[0] += a[0]
        avg_ans[1] += a[1]
        avg_ans[2] += b


    print('F1:\t', print_ans[0])
    print('Accuracy:\t', print_ans[1])
    print('Time:\t', print_ans[2])
    print(np.round(avg_ans[0] / t_num, 6), np.round(avg_ans[1] / t_num, 6), np.round(avg_ans[2] / t_num, 6))
