import torch
import torch.nn as nn


class VGGModel(nn.Module):
    """
    Model designed to perform a 2d convolution on the essays 
    with their activity encoded as an enum.

    Uses VGG (multiple smaller convolutions between max pooling) as
    per the description in the D2L.ai book. Applies padding so that
    the individual convolution layers don't affect the size.
    """

    def __init__(self, truncate_columns=False, c1=3, c2=7, c3=9, pool1=10, lin1=50):
        super(VGGModel, self).__init__()

        # optional extension gets rid of the cursor_position and
        # word_count columns as part of preprocessing
        self.truncate_columns = truncate_columns
        initial_columns = 5
        if truncate_columns:
            initial_columns = 3

        self.convolutional_layers = nn.Sequential(
            nn.BatchNorm2d(1),
            
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5, 1)),
            
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5, 1)),

            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5, initial_columns)),

            # this layer is important for getting the arrays all to the 
            # same size; because zero-padding works on a batch level, 
            # the different batches might have different sizes for their 
            # input arrays
            nn.AdaptiveMaxPool2d((pool1, 1)),

            # flatten the height and width dimensions
            nn.Flatten(start_dim=1, end_dim=3),
        )
    
        self.linear_layers = nn.Sequential(
            nn.Linear(pool1*c3, lin1),
            nn.Linear(lin1, 1),
        )
        

    def forward(self, x):
        # run the convolutional part of the network
        for layer in self.convolutional_layers:
            x = layer(x)
            #print(layer.__class__.__name__, 'output shape:\t', x.shape)

        # run the final linear part of the network
        for layer in self.linear_layers:
            x = layer(x)
            #print(layer.__class__.__name__, 'output shape:\t', x.shape)

        return x
    
    def preprocess_fn(self, essay_batch):
        # get rid of the final two columns if necessary
        if self.truncate_columns:
            essay_batch = torch.narrow(essay_batch, 2, 0, 3)

        # add in a dimension representing a single channel, necessary
        # for the convolution
        return torch.unsqueeze(essay_batch, 1)
    
    @classmethod
    def hyper_param_search(self):
        return {
            "c1": range(2, 6, 2),
            "c2": range(6, 12, 2),
            "c3": range(12, 18, 2),
            "pool1": range(10, 50, 10),
            "lin1": range(60, 90, 10),
        }