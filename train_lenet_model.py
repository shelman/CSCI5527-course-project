import itertools

import torch
from torch.utils.data import DataLoader
from train_test_fns import test_model, train_model

from models.lenet_inspired_model import LeNetInspiredModel
from split_dataset import SplitDataset
from utils import collate_fn


def create_dataloader(logs_folder, scores_file):
    dataset = SplitDataset(logs_folder, scores_file)
    return DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=collate_fn)


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    
    results = dict()
    hyper_params = LeNetInspiredModel.hyper_param_search()
    for c1 in hyper_params["c1"]:
        for c2 in hyper_params["c2"]:
            for pool in hyper_params["pool"]:
                for lin1 in hyper_params["lin1"]:
                    hp = dict(
                        c1=c1, 
                        c2=c2, 
                        pool=pool, 
                        lin1=lin1
                    )
                    model = LeNetInspiredModel(truncate_columns=False, **hp)

                    training_loader = create_dataloader(
                        "./data/preprocessed/train_logs_split_enum",
                        "./data/train_scores.csv",
                    )

                    model = train_model(model, training_loader, epochs=10, device=device, 
                                        print_progress=False)

                    test_loader = create_dataloader(
                        "./data/preprocessed/test_logs_split_enum",
                        "./data/test_scores.csv",
                    )

                    test_loss = test_model(model, test_loader, device=device)
                    results[str(hp)] = test_loss

    results_sorted = sorted(results.items(), key=lambda item: item[1])
    for result in results_sorted:
        print(f"{result[0]}: {result[1]}")

if __name__ == "__main__":
    main()