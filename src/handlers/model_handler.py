import os
import sys
import logging

import torch

class ModelHandler(object):

    def __init__(self, path):
        super(ModelHandler, self).__init__()
        self.path = path
    
    def not_exist(self):
        not_exist = not os.path.exists(self.path)
        if not_exist:
            logging.debug('Init model not exist!')
        return not_exist 

    def load(self, model_name):
        model_dict = torch.load(os.path.join(self.path, model_name))
        return model_dict 

    def save(self, model_dict, model_name):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        torch.save(model_dict, os.path.join(self.path, model_name))