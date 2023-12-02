import torch
import torch.nn as nn


class LeNetInspiredModel(nn.Module):
    """
    Model designed to perform a 2d convolution on the essays 
    with their activity encoded as an enum.

    Architecture is inspired by the original LeNet from the 
    1998 paper.
    """

    def __init__(self):
        super(LeNetInspiredModel, self).__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm2d(1),
            
            nn.Conv2d(1, 5, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            
            nn.Conv2d(5, 10, kernel_size=(5, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            
            nn.Conv2d(10, 5, kernel_size=(5, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(5, 1, kernel_size=(5, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            # this layer is important for getting the arrays all to the 
            # same size; because zero-padding works on a batch level, 
            # the different batches might have different sizes for their 
            # input arrays
            nn.AdaptiveAvgPool2d((1000, 1)),
            
            nn.Flatten(),
            
            nn.Linear(1000, 100),
            nn.ReLU(),
            
            nn.Linear(100, 1),
        )

    def forward(self, x):
        return self.layers(x)
    
    def preprocess_fn(self, essay_batch):
        # add in a dimension representing a single channel, necessary
        # for the convolution
        return torch.unsqueeze(essay_batch, 1)