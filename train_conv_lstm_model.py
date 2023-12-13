import torch
from torch.utils.data import DataLoader
from train_test_fns import test_model, train_model

from models.conv_lstm_model import ConvLSTMModel
from split_dataset import SplitDataset
from utils import collate_fn


def create_dataloader(logs_folder, scores_file):
    dataset = SplitDataset(logs_folder, scores_file)
    return DataLoader(dataset, batch_size=200, shuffle=True, collate_fn=collate_fn)


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = ConvLSTMModel(truncate_columns=False)

    training_loader = create_dataloader(
        "./data/preprocessed/train_logs_split_enum",
        "./data/train_scores.csv",
    )

    model = train_model(model, training_loader, epochs=30, device=device)

    test_loader = create_dataloader(
        "./data/preprocessed/test_logs_split_enum",
        "./data/test_scores.csv",
    )

    test_model(model, test_loader, device=device)

if __name__ == "__main__":
    main()