import pandas as pd
import torch
from torch.utils.data import Dataset

from pathlib import Path


class SplitDataset(Dataset):
    """
    Dataset representing a folder with split essay files in it,
    one csv file per essay.

    Relies on the presence of a scores file as well, which does not need
    to be preprocessed from that provided by the competition.
    """
    def __init__(self, logs_folder, scores_file):
        self.essay_files = list(Path(logs_folder).glob("*.csv"))
        self.file_count = len(self.essay_files)

        self.scores = pd.read_csv(scores_file)
        self.scores.set_index("id")

    def __len__(self):
        return self.file_count

    def __getitem__(self, idx):
        # pull out the essay
        essay_file = self.essay_files[idx]
        essay = pd.read_csv(essay_file)
        
        # retrieve the relevant score. assumes that the log file for each
        # essay is titled <essay id>.csv
        essay_id = Path(essay_file).stem
        score = self.scores.loc[self.scores["id"] == essay_id, "score"].values[0]

        return torch.tensor(essay.to_numpy()), torch.tensor(score)