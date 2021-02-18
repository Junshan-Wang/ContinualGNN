import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd

import numpy as np
import logging


class EWC(nn.Module):

    def __init__(self, model, ewc_lambda = 0, ewc_type = 'ewc'):
        super(EWC, self).__init__()
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.ewc_type = ewc_type

    def _update_mean_params(self):
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.register_buffer(_buff_param_name + '_estimated_mean', param.data.clone())

    def _update_fisher_params(self, nodes, labels):
        log_likelihood = self.model.loss(nodes, labels)
        grad_log_liklihood = autograd.grad(log_likelihood, self.model.parameters())
        _buff_param_names = [param[0].replace('.', '__') for param in self.model.named_parameters()]
        for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
            self.register_buffer(_buff_param_name + '_estimated_fisher', param.data.clone() ** 2)

    def _save_fisher_params(self):
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            estimated_mean = getattr(self, '{}_estimated_mean'.format(_buff_param_name))
            estimated_fisher = np.array(getattr(self, '{}_estimated_fisher'.format(_buff_param_name)))
            np.savetxt('estimated_mean', estimated_mean)
            np.savetxt('estimated_fisher', estimated_fisher)
            print(np.mean(estimated_fisher), np.max(estimated_fisher), np.min(estimated_fisher))
            break


    def register_ewc_params(self, nodes, labels):
        self._update_fisher_params(nodes, labels)
        self._update_mean_params()

    def _compute_consolidation_loss(self):
        losses = []
        for param_name, param in self.model.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            estimated_mean = getattr(self, '{}_estimated_mean'.format(_buff_param_name))
            estimated_fisher = getattr(self, '{}_estimated_fisher'.format(_buff_param_name))
            if self.ewc_type == 'l2':
                losses.append((10e-6 * (param - estimated_mean) ** 2).sum())
            else:
                losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
        return 1 * (self.ewc_lambda / 2) * sum(losses)

    def loss(self, nodes, labels):
        loss1 = self.model.loss(nodes, labels)
        loss2 = self._compute_consolidation_loss()
        loss = loss1 + loss2
        return loss

    def save(self, filename):
        torch.save(self.model, filename)

    def load(self, filename):
        self.model = torch.load(filename)
