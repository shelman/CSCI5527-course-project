import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

def get_data():
    # train logs contains about 5000 logs without labels
    train_logs = pd.read_csv("./linking-writing-processes-to-writing-quality/train_logs.csv")
    # train scores only contain labels for each id
    train_scores = pd.read_csv("./linking-writing-processes-to-writing-quality/train_scores.csv")
    test_logs = pd.read_csv("./linking-writing-processes-to-writing-quality/test_logs.csv")

    return train_logs, train_scores, test_logs

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

def preprocess(train_logs, train_scores, test_logs):
    # set index using id
    train_scores = train_scores.set_index(["id"])

    # one hot encode the activity attribute 
    categories_in_activity = ["Nonproduction", "Input", "Remove/Cut", "Replace", "Paste"]
    train_logs.loc[~train_logs["activity"].isin(categories_in_activity), "activity"] = "Move"
    one_hot_encoder = pd.get_dummies(train_logs["activity"], dtype=int)
    train_logs = train_logs.drop(columns=["activity"])
    train_logs = pd.concat([train_logs, one_hot_encoder], axis=1)

    train_logs["text_change"] = train_logs["text_change"].apply(count_char)

    # remove categorical attributes
    train_logs = train_logs.drop(columns=["down_event", "up_event"])

    # set index using id attribute
    train_logs = train_logs.set_index(["id"])

    # print(train_logs.head(10))
    # print(train_logs.info())
    return train_logs, train_scores, test_logs


class TrainingDataset(Dataset):
    def __init__(self, data, labels):
        self.data = self._generate_senquence_(data)
        self.labels = labels.values

    def _generate_senquence_(self, data):
        sequences_data = []
        grouped_data = data.groupby(["id"])
        for _, x in grouped_data:
            sequences_data.append(x.values)
        return sequences_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx])
        y =  torch.tensor(self.labels[idx])
        return x, y

def create_training_dataloader():
    train_logs, train_scores, test_logs = get_data()

    train_logs, train_scores, test_logs = preprocess(train_logs, train_scores, test_logs)

    training_dataset = TrainingDataset(train_logs, train_scores)
    training_loader = DataLoader(training_dataset, batch_size=32, shuffle=True) # need to implement collate_fn=pad_sequences
    return training_loader, len(training_dataset)
