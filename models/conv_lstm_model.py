import torch
import torch.nn as nn


class ConvLSTMModel(nn.Module):
    """
    Model designed to perform a 2d convolution on the essays 
    with their activity encoded as an enum, followed by an
    LSTM.

    Uses VGG (multiple smaller convolutions between max pooling) as
    per the description in the D2L.ai book. Applies padding so that
    the individual convolution layers don't affect the size.
    """

    def __init__(self, truncate_columns=False, channels=16, final_pool=10, lstm_hidden=64):
        super(ConvLSTMModel, self).__init__()

        # optional extension gets rid of the cursor_position and
        # word_count columns as part of preprocessing
        self.truncate_columns = truncate_columns
        initial_columns = 5
        if truncate_columns:
            initial_columns = 3

        self.convolutional_layers = nn.Sequential(
            nn.BatchNorm2d(1),

            nn.Conv2d(1, int(channels/2), kernel_size=(15, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5, 1)),
            
            nn.Conv2d(int(channels/2), int(channels), kernel_size=(15, initial_columns)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5, 1)),

            # this layer is important for getting the arrays all to the 
            # same size; because zero-padding works on a batch level, 
            # the different batches might have different sizes for their 
            # input arrays
            nn.AdaptiveMaxPool2d((final_pool, 1)),

            # flatten the height and width dimensions
            nn.Flatten(start_dim=2, end_dim=3),
        )

        self.lstm = nn.LSTM(channels, lstm_hidden, batch_first=True)
        self.post_lstm_flatten = nn.Flatten(start_dim=1, end_dim=2)

        mid_linear = int(lstm_hidden*final_pool/3)
        self.linear_layers = nn.Sequential(
            nn.Linear(lstm_hidden*final_pool, mid_linear),
            nn.Linear(mid_linear, 1),
        )
        

    def forward(self, x):
        # run the convolutional part of the network
        for layer in self.convolutional_layers:
            x = layer(x)
            #print(layer.__class__.__name__, 'output shape:\t', x.shape)

        # move the channel dimension to the end to prepare for the lstm
        x = x.permute(0, 2, 1)
        #print('Permute output shape:\t', x.shape)
        
        x, (h_n, c_n) = self.lstm(x)
        #print('LSTM output shape:\t', x.shape)

        x = self.post_lstm_flatten(x)
        #print('Post-LSTM flatten output shape:\t', x.shape)

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
            "channels": range(10, 32, 4),
            "final_pool": range(10, 100, 10),
            "lstm_hidden": range(64, 204, 20),
        }