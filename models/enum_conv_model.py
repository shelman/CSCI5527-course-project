import torch
import torch.nn as nn


class EnumConvModel(nn.Module):
    """
    Model designed to perform a 2d convolution on the essays 
    with their activity encoded as an enum.
    """

    def __init__(self):
        super(EnumConvModel, self).__init__()

        self.pre_linear_layers = [
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 1, kernel_size=3),
            #nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(1, 1, kernel_size=2),

            # this layer is important for getting the arrays all to the 
            # same size; because zero-padding works on a batch level, 
            # the different batches might have different sizes for their 
            # input arrays
            nn.AdaptiveAvgPool2d((2000, 5)),
        ]

        self.lin1 = nn.Linear(2000*5, 100)
        self.output = nn.Linear(100, 1)

    def forward(self, x):
        for layer in self.pre_linear_layers:
            x = layer(x)

        # get rid of the channel dimension
        x = torch.squeeze(x, dim=1)
        # flatten the height and width of the array
        x = torch.flatten(x, start_dim=1, end_dim=2)

        x = nn.functional.leaky_relu(self.lin1(x))
        x = self.output(x)
        return x
    
    def preprocess_fn(self, essay_batch):
        # add in a dimension representing a single channel, necessary
        # for the convolution
        return torch.unsqueeze(essay_batch, 1)