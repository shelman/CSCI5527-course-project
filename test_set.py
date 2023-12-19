import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


def count_char(input_str):
    '''
    Convert text_change into numerical values

    Substitue with the string length depending on the type of characters in the string
    '''

    if input_str == "NoChange":
        return 0
    
    # Check if the string contains only letters
    if input_str.isalpha():
        return len(input_str)
    
    # Check if the string contains special characters or symbols
    else:
        count = 0
        if input_str == " ":
            return count
        
        for char in input_str:
            if char.isalnum():
                count += 1
        return count



def get_test_data():
    test_logs = pd.read_csv("./dataset/mytest.csv")

    categories_in_activity = ["Nonproduction", "Input", "Remove/Cut", "Replace", "Paste"]
    test_logs.loc[~test_logs["activity"].isin(categories_in_activity), "activity"] = "Move"
    one_hot_encoder = pd.get_dummies(test_logs["activity"], dtype=int)
    # categories_in_activity.append("Move")
    # for c in categories_in_activity:
    #     if c not in one_hot_encoder.columns:
    #         one_hot_encoder[c] = 0
    test_logs = test_logs.drop(columns=["activity"])
    test_logs = pd.concat([test_logs, one_hot_encoder], axis=1)

    test_logs["text_change"] = test_logs["text_change"].apply(count_char)

    test_logs = test_logs.drop(columns=["down_event", "up_event"])

    time_gap = []
    # gap_pct = []

    for _, group in test_logs.groupby("id"):
        diff = group["down_time"] - group["up_time"].shift(1).fillna(0)
        time_gap.append(diff)
        # gap_pct.append(diff / diff.sum())

    pause = pd.concat(time_gap, axis=0)
    pause.name = "pause"
    # pause_pct = pd.concat(gap_pct, axis=0)
    # pause_pct.name = "pause_pct"

    test_logs = pd.concat([test_logs, pause], axis=1)

    test_logs = test_logs.set_index(["id"])
    
    index = []
    mean_pause = []
    del_num = []
    add_num = []
    del_len = []
    add_len = []
    del_ratio = []
    add_ratio = []
    product_process_ratio = []

    for idx, group in test_logs.groupby("id"):
        index.append(idx)

        mean_pause.append(group["pause"].mean())

        remove = group["Remove/Cut"] == 1
        input = group["Input"] == 1
        del_num.append(remove.sum())

        add_num.append(input.sum() + (group["Paste"] == 1).sum())

        remove_char_sum = group.loc[remove, "text_change"].sum()
        add_char_sum = group.loc[input, "text_change"].sum()
        del_len.append(remove_char_sum)
        add_len.append(add_char_sum)

        total_time = group.iloc[-1]["up_time"]
        del_time = group.loc[remove, "action_time"].sum()
        add_time = group.loc[input, "action_time"].sum()
        del_ratio.append(del_time)
        add_ratio.append(add_time)

        total_char = add_char_sum
        result_char = add_char_sum - remove_char_sum
        product_process_ratio.append(result_char / total_char)

    test_essays = pd.DataFrame({"mean_pause": mean_pause, 
                                "del_num": del_num,
                                "insert_num": add_num, 
                                "del_length": del_len, 
                                "insert_length": add_len, 
                                "del_ratio": del_ratio, 
                                "insert_ratio": add_ratio, 
                                "product_process_ratio": product_process_ratio}, index=index
                                )

    return test_logs, test_essays

class TestDataset(Dataset):
    def __init__(self, data, extras):
        self.data = self._generate_sequence_(data)
        self.ids = data["id"]
        self.extras = extras.values

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
        ids = self.ids[idx]
        extras = torch.tensor(self.extras[idx])
        return data, ids, extras

def create_test_dataloader():
    test_logs, test_essays = get_test_data()

    # test_logs.to_csv("mytest_logs.csv")
    # test_essays.to_csv("mytest_essays.csv")

    test_dataset = TestDataset(test_logs, test_essays)

    # Now you can create DataLoader instances for training and validation
    test_loader = DataLoader(test_dataset, batch_size=1)

    return test_loader