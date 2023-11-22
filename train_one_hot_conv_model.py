import torch
from torch.utils.data import DataLoader
from train_test_fns import test_model, train_model

from models.one_hot_conv_model import OneHotConvModel
from split_dataset import SplitDataset
from utils import collate_fn


def create_dataloader(logs_folder, scores_file):
    dataset = SplitDataset(logs_folder, scores_file)
    return DataLoader(dataset, batch_size=40, shuffle=True, collate_fn=collate_fn)


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = OneHotConvModel()

    training_loader = create_dataloader(
        "./data/preprocessed/train_logs_split_one_hot",
        "./data/train_scores.csv",
    )

    model = train_model(model, training_loader, epochs=10, device=device)

    test_loader = create_dataloader(
        "./data/preprocessed/test_logs_split_one_hot",
        "./data/test_scores.csv",
    )

    test_model(model, test_loader, device=device)

if __name__ == "__main__":
    main()