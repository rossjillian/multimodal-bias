import torch
import torch.nn as nn
import torch.nn.functional as f


class COCO10Classifier(nn.Sequential):
    def __init__(self, in_size):
        super(COCO10Classifier, self).__init__()
        self.dense = nn.Linear(in_size, 10)

    def forward(self, x):
        x = self.dense(x)
        return x

