import torch

class Aggregator(object):

    def __init__(self):
        super(Aggregator, self).__init__()
    
    def aggregate(self, mask, features):
        aggregate_features = torch.matmul(mask, features)
        return aggregate_features