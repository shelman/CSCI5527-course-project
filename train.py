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

training_loader, sample_size = create_training_dataloader()

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        x = self.fc(output[:, 0, :])
        return x

# Define input size, hidden layer size, and output size
input_size = 13
hidden_size = 128
output_size = 12

# # Instantiate the model
model = LSTM(input_size, hidden_size, output_size).to(device)

print(model)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# training dataset
epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    print("-" * 10)
    running_loss = 0
    running_corrects = 0
    for images, labels in training_loader:
        # move to the specified device
        images, labels = images.to(device), labels.to(device)

        # forward
        optimizer.zero_grad()
        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    # statistics
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / sample_size
    epoch_acc = running_corrects.double() / sample_size

    print(f"Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
