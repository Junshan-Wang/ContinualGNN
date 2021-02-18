import argparse
import logging
import numpy as np
import random
import logging

import torch

def parse_argument():
    parser = argparse.ArgumentParser(description = 'pytorch version of GraphSAGE')
    # data options
    parser.add_argument('--data', type = str, default = 'cora')
    
    parser.add_argument('--num_epochs', type = int, default = 100)
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--seed', type = int, default = 13)
    parser.add_argument('--cuda', action = 'store_true', help = 'use CUDA')
    parser.add_argument('--num_neg_samples', type = int, default = 10)
    parser.add_argument('--num_layers', type = int, default = 2)
    parser.add_argument('--embed_size', type = int, default = 64)
    parser.add_argument('--learning_rate', type = float, default = 0.1)

    parser.add_argument('--detect_strategy', type = str, default = 'bfs')  # 'simple' / 'bfs'
    parser.add_argument('--new_ratio', type = float, default = 0.0)

    parser.add_argument('--memory_size', type = int, default = 0)
    parser.add_argument('--memory_strategy', type = str, default = 'class')   # 'random' / 'class'
    parser.add_argument('--p', type = float, default = 1)
    parser.add_argument('--alpha', type = float, default = 0.0)
    
    parser.add_argument('--ewc_lambda', type = float, default = 0.0)
    parser.add_argument('--ewc_type', type = str, default = 'ewc')  # 'l2' / 'ewc'

    parser.add_argument('--eval', action = 'store_true')

    args = parser.parse_args()

    return args

def print_args(args):
    config_str = 'Parameters: '
    for name, value in vars(args).items():
        config_str += str(name) + ': ' + str(value) + '; '
    logging.info(config_str)


def check_device(cuda):
    if torch.cuda.is_available():
        if not cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            device_id = torch.cuda.current_device()
            print('using device', device_id, torch.cuda.get_device_name(device_id))
    device = torch.device("cuda" if cuda else "cpu")
    logging.info('Device:' + str(device))
    return device



def node_classification(trut, pred, name = ''):
    from sklearn import metrics
    f1 = np.round(metrics.f1_score(trut, pred, average="macro"), 6)
    acc = np.round(metrics.f1_score(trut, pred, average="micro"), 6)
    logging.info(name + '   Macro F1:' +  str(f1) \
            + ";    Micro F1:" +  str(acc))
    return f1, acc
