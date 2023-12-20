import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from utils_hybrid import create_train_dataloader


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

batch_size = 32
train_loader, val_loader = create_train_dataloader(batch_size)

class myNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, extra_features_size):
        super(myNN, self).__init__()
        self.conv1 = nn.Conv2d(1, input_size, kernel_size=(5, input_size))
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(5,1))
        #self.adaptivepool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2)
        self.layernorm = nn.LayerNorm(hidden_size + extra_features_size)
        self.batchnorm = nn.BatchNorm1d(hidden_size + extra_features_size)
        self.instancenorm = nn.InstanceNorm1d(hidden_size + extra_features_size, affine=True)
        self.fc = nn.Linear(hidden_size + extra_features_size, output_size)

    def forward(self, input, extra_features):
        input = torch.unsqueeze(input, dim=1)
        x = self.conv1(input)
        x = self.relu1(x)
        x = self.pool1(x)
        x = torch.squeeze(x, dim=-1)
        x = x.permute(2, 0, 1)
        #x = self.adaptivepool(x)

        output, (h_n, c_n) = self.lstm(x)
        #h_n.view(batch_size,-1)
        
        # Concatenate the LSTM output with additional features
        l = output.size(0)
        extra_features = extra_features.unsqueeze(0).repeat(l, 1, 1)
        x = torch.cat((output, extra_features), dim=2).permute(1, 2, 0)
        x = self.batchnorm(x)
        x = torch.mean(x.permute(2, 0, 1), dim=0)
        # x = self.instancenorm(x)
        # x = torch.max(x.permute(2, 0, 1), dim=0)[0]
        
        # x = torch.cat((h_n[-1,:,:], extra_features), dim=1)
        # x = torch.cat((c_n[-1,:,:], extra_features), dim=1)

        # Pass the combined features through the fully connected layer
        # x = self.layernorm(x)
        # x = self.batchnorm(x)
        x = self.fc(x)
        return x

# Define input size, hidden layer size, and output size
input_size = len(train_loader.dataset[0][0][0])
hidden_size = 32
output_size = 1
extra_feature_size = len(train_loader.dataset[0][2])

# # Instantiate the model
model = myNN(input_size, hidden_size, output_size, extra_feature_size).to(device)

print(model)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

epochs = 100
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    model.train()  # Set the model to train mode
    running_loss = 0
    for data, labels, extras in train_loader:
        # move to the specified device
        data = data.float().to(device)
        labels = labels.float().to(device)
        extras = extras.float().to(device)

        # forward
        optimizer.zero_grad()
        outputs = model(data, extras)
        # print(outputs)
        # print(labels)
        loss = torch.sqrt(criterion(outputs, labels))

        loss.backward()
        optimizer.step()

    # statistics
        running_loss += loss.item() * data.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)

    print(f"Training Loss: {epoch_loss:.4f}")
    
    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_running_loss = 0
    with torch.no_grad():  # Disable gradient computation during validation
        for val_data, val_labels, val_extras in val_loader:
            val_data = val_data.float().to(device)
            val_labels = val_labels.float().to(device)
            val_extras = val_extras.float().to(device)

            val_outputs = model(val_data, val_extras)
            val_loss = torch.sqrt(criterion(val_outputs, val_labels))

            val_running_loss += val_loss.item() * val_data.size(0)

    val_epoch_loss = val_running_loss / len(val_loader.dataset)
    
    print(f"Validation Loss: {val_epoch_loss:.4f}")
    print("-" * 10)