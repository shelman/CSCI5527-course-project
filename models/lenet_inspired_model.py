import torch
import torch.nn as nn


class LeNetInspiredModel(nn.Module):
    """
    Model designed to perform a 2d convolution on the essays 
    with their activity encoded as an enum.

    Architecture is inspired by the original LeNet from the 
    1998 paper.
    """

    def __init__(self, truncate_columns=False):
        super(LeNetInspiredModel, self).__init__()

        # optional extension gets rid of the cursor_position and
        # word_count columns as part of preprocessing
        self.truncate_columns = truncate_columns
        initial_columns = 5
        if truncate_columns:
            initial_columns = 3

        self.layers = nn.Sequential(
            nn.BatchNorm2d(1),
            
            nn.Conv2d(1, 10, kernel_size=(5, initial_columns)),
            nn.MaxPool2d(kernel_size=(5, 1)),
            nn.ReLU(),
            
            nn.Conv2d(10, 16, kernel_size=(5, 1)),
            nn.MaxPool2d(kernel_size=(5, 1)),
            nn.ReLU(),

            # this layer is important for getting the arrays all to the 
            # same size; because zero-padding works on a batch level, 
            # the different batches might have different sizes for their 
            # input arrays
            nn.AdaptiveMaxPool2d((10, 1)),
            
            # flatten the channel, height and width dimensions
            nn.Flatten(start_dim=1, end_dim=3),
            
            nn.Linear(160, 50),
            nn.ReLU(),
            
            nn.Linear(50, 1),
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            #print(layer.__class__.__name__, 'output shape:\t', x.shape)

        print('Output shape:\t', x.shape)
        return x
    
    def preprocess_fn(self, essay_batch):
        # get rid of the final two columns if necessary
        if self.truncate_columns:
            essay_batch = torch.narrow(essay_batch, 2, 0, 3)

        # add in a dimension representing a single channel, necessary
        # for the convolution
        return torch.unsqueeze(essay_batch, 1)