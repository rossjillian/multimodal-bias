import torch
import torch.nn as nn
import torch.nn.functional as f


class COCO10Classifier(nn.Sequential):
    def __init__(self):
        super(COCO10Classifier, self).__init__()
        self.dense = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.Softmax(self.dense(x))
        return x

