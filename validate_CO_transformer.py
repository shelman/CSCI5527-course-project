import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils_trans import create_training_dataloader
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification, CamembertModel, get_linear_schedule_with_warmup, RobertaConfig, RobertaTokenizer, RobertaModel


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(device)


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

batch_size = 1

training_loader, validation_loader, sample_size = create_training_dataloader(batch_size=batch_size)


class CustomRobertaForRegression(nn.Module):
    def __init__(self, model_name='roberta-base', num_labels=1, num_layers_to_freeze=8):
        super(CustomRobertaForRegression, self).__init__()
        
        # Load RoBERTa configuration
        config = RobertaConfig.from_pretrained(model_name)

        
        # Load pre-trained RoBERTa model
        self.roberta = RobertaModel.from_pretrained(model_name, config=config)
        
        # Iterate through the layers and freeze/unfreeze accordingly
        for param in self.roberta.parameters():
            param.requires_grad = False

        # Modify the model's head for regression
        self.roberta.config.num_labels = num_labels
        #self.roberta.pooler.dense = nn.Linear(config.hidden_size, num_labels)
        self.regressor = nn.Linear(config.hidden_size, 1, dtype=torch.float32)  # Assuming BERT's hidden size is 768
        self.roberta.pooler.activation = nn.Identity()

        print(self.roberta)


    def forward(self, input_ids, attention_mask, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, head_mask=head_mask)
        lhs_output = outputs.last_hidden_state
        logits = self.regressor(lhs_output)
        return logits

        #outputs = self.roberta(input_ids, attention_mask=attention_mask, head_mask=head_mask)
        #pooled_output = outputs.pooler_output
        #logits = self.regressor(pooled_output)
        return logits

# Define input size, hidden layer size, and output size
hidden_channels = 32
hidden_size = 768
output_size = 1


# Instantiate the model, optimizer, and loss function
model = CustomRobertaForRegression().to(device)
criterion = nn.MSELoss()

# Load weights from a .pth file
checkpoint_path = 'RoBERTa_lhs.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))  # Specify map_location based on your device

# Load the model state_dict from the checkpoint
model.load_state_dict(checkpoint)


# Define window size and overlap
window_size = 700


model.eval()
val_loss = 0

for batch in validation_loader:
    input_ids, targets = batch

    # Concatenate values across columns with spaces
    result_string = ' '.join([' '.join(map(str, arr)) for arr in input_ids])
    characters_to_remove = '\[]"\''
    translation_table = str.maketrans("", "", characters_to_remove)
    result_string = result_string.translate(translation_table)

    result_list = []
    sample_loss = 0
    # Split the long text into 500-character increments
    for i in range(0, len(result_string), window_size):
        segment = result_string[i:i+window_size]

        # Tokenize the segment
        tokens = tokenizer(segment, padding='max_length', return_tensors='pt')

        input_ids = tokens['input_ids']
        attn_masks = tokens['attention_mask']
        input_ids = input_ids.to(device)
        attn_masks = attn_masks.to(device)

        outputs = model(input_ids, attn_masks)
        aggregated_logits = torch.mean(outputs, dim=1, keepdim=False)

        result_list.append(aggregated_logits.squeeze().item())

    mean_results = sum(result_list)/len(result_list)
    sample_loss = (mean_results-targets.item())**2
    print('Sample Loss:{:.3f} | Predicted Score:{:.3f}'.format(sample_loss, mean_results))
    val_loss += sample_loss

average_val_loss = val_loss / len(validation_loader)
print(f'Validation Loss: {average_val_loss}')