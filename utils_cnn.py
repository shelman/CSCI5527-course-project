import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


def get_train_data():
    # train logs contains about 5000 logs without labels
    train_logs = pd.read_csv("./dataset/new_train_logs.csv", index_col=0)
    # train scores only contain labels for each id
    train_scores = pd.read_csv("./dataset/new_train_scores.csv", index_col=0)
    
    return train_logs, train_scores



class TrainingDataset(Dataset):
    def __init__(self, data, labels):
        self.data = self._generate_sequence_(data)
        self.labels = labels.values

    def _generate_sequence_(self, data):
        sequences_data = []
        grouped_data = data.groupby(["id"])
        for _, x in grouped_data:
            sequences_data.append(x.values)
        return sequences_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx])
        labels = torch.tensor(self.labels[idx])
        return data, labels

def collate_fn(batch):
    sequences, labels = zip(*batch)
    max_length = max(len(seq) for seq in sequences)
    padded = [padding(seq, max_length) for seq in sequences]
    return torch.stack(padded), torch.stack(labels)

def padding(sequence, max_length):
    pad_zeros = torch.zeros([max_length - len(sequence), sequence.shape[1]], dtype=float)
    padded_seq = torch.cat((sequence, pad_zeros))
    return padded_seq

def create_train_dataloader(batch_size):
    train_logs, train_scores = get_train_data()

    train_dataset = TrainingDataset(train_logs, train_scores)

    # Define the sizes for training and validation sets
    dataset_size = len(train_dataset)
    train_size = int(0.8 * dataset_size)  # 80% for training
    val_size = dataset_size - train_size  # 20% for validation

    # Use random_split to create training and validation datasets
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Now you can create DataLoader instances for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader