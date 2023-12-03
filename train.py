import torch
import torch.nn as nn
import torch.optim as optim

from utils import create_training_dataloader

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

batch_size = 32
train_loader, val_loader, train_size = create_training_dataloader(batch_size)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, extra_features_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size + extra_features_size, output_size)

    def forward(self, x, extra_features):
        output, _ = self.lstm(x)

        # Concatenate the LSTM output with additional features
        x = torch.cat((output[:, 0, :], extra_features), dim=1)

        # Pass the combined features through the fully connected layer
        x = self.fc(x)
        return x

# Define input size, hidden layer size, and output size
input_size = 15
hidden_size = 128
output_size = 1
extra_feature_size = 5

# # Instantiate the model
model = LSTM(input_size, hidden_size, output_size, extra_feature_size).to(device)

print(model)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# training dataset
epochs = 25
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    model.train()  # Set the model to train mode
    running_loss = 0
    for images, labels, extras in train_loader:
        # move to the specified device
        images = images.float().to(device)
        labels = labels.float().to(device)
        extras = extras.float().to(device)

        # forward
        optimizer.zero_grad()
        outputs = model(images, extras)
        # print(outputs)
        # print(labels)
        loss = torch.sqrt(criterion(outputs, labels))

        loss.backward()
        optimizer.step()

    # statistics
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / train_size

    print(f"Training Loss: {epoch_loss:.4f}")
    
    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_running_loss = 0
    with torch.no_grad():  # Disable gradient computation during validation
        for val_images, val_labels, val_extras in val_loader:
            val_images = val_images.float().to(device)
            val_labels = val_labels.float().to(device)
            val_extras = val_extras.float().to(device)

            val_outputs = model(val_images, val_extras)
            val_loss = torch.sqrt(criterion(val_outputs, val_labels))

            val_running_loss += val_loss.item() * val_images.size(0)

    val_epoch_loss = val_running_loss / len(val_loader.dataset)
    
    print(f"Validation Loss: {val_epoch_loss:.4f}")
    print("-" * 10)