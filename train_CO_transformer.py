import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils_trans import create_training_dataloader
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification, CamembertModel, get_linear_schedule_with_warmup



device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

device = "cpu"


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

batch_size = 1

training_loader, validation_loader, sample_size = create_training_dataloader(batch_size=batch_size)


# Define a simple regression model using BERT as a feature extractor
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        #self.conv1 = nn.Conv2d(1, out_channels=hidden_channels, kernel_size=(10, 13))
        #self.relu1 = nn.LeakyReLU()
        #self.pool1 = nn.MaxPool2d(kernel_size=(10,1))
        #self.conv2 = nn.Conv2d(hidden_channels, out_channels=hidden_channels, kernel_size=(10, 1))
        #self.relu2 = nn.LeakyReLU()
        #self.pool2 = nn.MaxPool2d(kernel_size=(10,1))
        #self.conv3 = nn.Conv2d(hidden_channels, out_channels=2, kernel_size=(10, 1))
        #self.relu3 = nn.LeakyReLU()
        #self.pool3 = nn.MaxPool2d(kernel_size=(10,1))
        #self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.bert = CamembertModel.from_pretrained('camembert-base')
        self.regressor = nn.Linear(hidden_size, 1, dtype=torch.float32)  # Assuming BERT's hidden size is 768

    def forward(self, input_ids, attention_mask):
        #input = torch.unsqueeze(input_ids, attention_mask dim=1)
        #x = self.conv1(input)
        #x = self.relu1(x)
        #x = self.pool1(x)
        #x = self.conv2(x)
        #x = self.relu2(x)
        #x = self.pool2(x)
        #x = self.conv3(x)
        #x = self.relu3(x)
        #x = self.pool3(x)
        #x = torch.squeeze(x, dim=-1)
        #x = x.permute(0, 2, 1)
        #x = x.view(-1, 1)

        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.regressor(pooled_output)
        return logits

# Define input size, hidden layer size, and output size
hidden_channels = 32
hidden_size = 768
output_size = 1


# Instantiate the model, optimizer, and loss function
model = RegressionModel().to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.MSELoss()


# Define window size and overlap
window_size = 700
overlap = 100


epochs = 5
total_steps = len(training_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,       
                 num_warmup_steps=0, num_training_steps=total_steps)


for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in training_loader:
        input_ids, targets = batch

        # Concatenate values across columns with spaces
        result_string = ' '.join([' '.join(map(str, arr)) for arr in input_ids])

        # Initialize a list to store tokenized results
        input_ids = []
        attn_masks = []

        # Split the long text into 500-character increments
        for i in range(0, len(result_string), window_size):
            if np.random.uniform() < .05:
                segment = result_string[i:i+window_size]
                
                # Tokenize the segment
                tokens = tokenizer(segment, padding='max_length', return_tensors='pt')
            
                # Append the tokenized result to the list
                input_ids.append(tokens['input_ids'])
                attn_masks.append(tokens['attention_mask'])
            else:
                next

        targets = targets.repeat(len(input_ids), 1).to(dtype=torch.float32).to(device)
        input_ids = torch.cat(input_ids, dim=0).to(device)
        attn_masks = torch.cat(attn_masks, dim=0).to(device)

        input_ids = input_ids
        attn_masks = attn_masks

        outputs = model(input_ids, attn_masks)
        loss = criterion(outputs.squeeze(), targets.squeeze())

        loss = loss.to(dtype=torch.float32)
        print(loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    average_loss = total_loss / len(training_loader)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss}')
    '''
    # Validation loop
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in validation_loader:
            input_ids, attention_mask = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            targets = targets.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits.squeeze(), targets)
            val_loss += loss.item()

    average_val_loss = val_loss / len(validation_loader)
    print(f'Validation Loss: {average_val_loss}')
    '''










'''


for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    running_loss = 0
    for images, labels in training_loader:


        # move to the specified device
        images, labels = images.float().to(device), labels.float().to(device)

        # forward
        optimizer.zero_grad()
        outputs = model(images)
        # print(outputs)
        # print(labels)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    # statistics
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / sample_size

    print(f"Training Loss: {epoch_loss:.4f}")


    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_running_loss = 0
    with torch.no_grad():  # Disable gradient computation during validation
        for val_images, val_labels in validation_loader:
            val_images, val_labels = val_images.float().to(device), val_labels.float().to(device)

            val_outputs = model(val_images)
            val_loss = criterion(val_outputs, val_labels)

            val_running_loss += val_loss.item() * val_images.size(0)

    val_epoch_loss = val_running_loss / len(validation_loader.dataset)
    
    print(f"Validation Loss: {val_epoch_loss:.4f}")
    print("-" * 10)
'''