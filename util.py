import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# train logs contains about 5000 logs without labels
train_logs = pd.read_csv("./linking-writing-processes-to-writing-quality/train_logs.csv")
# train scores only contain labels for each id
train_scores = pd.read_csv("./linking-writing-processes-to-writing-quality/train_scores.csv")
test_logs = pd.read_csv("./linking-writing-processes-to-writing-quality/test_logs.csv")

count_id_list = list(train_logs.groupby(["id"]).count()["event_id"])
# correspond labels with the samples
train_scores = train_scores["score"].repeat(count_id_list).reset_index()


# convert the categorical feature text_change into numerical feature
# substitue with the string length depending on the type of characters in the string
def count_char(input_str):
    if input_str == "NoChange":
        return 0
    
    # Check if the string contains only letters
    if input_str.isalpha():
        return len(input_str)
    
    # Check if the string contains special characters or symbols
    else:
        count = 0
        for char in input_str:
            if char.isalnum():
                count += 1

        # return -1 when the string only contains 1 special character
        if count == 0:
            count = -1
        return count

train_logs["text_change"] = train_logs["text_change"].apply(count_char)





class TrainingDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx])
        label = torch.tensor(self.labels[idx])
        return data, label

training_dataset = TrainingDataset(train_logs, train_scores)
test_dataset = torch.tensor(test_logs.values)

training_loader = DataLoader(training_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

print(training_loader)
print(test_loader)