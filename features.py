import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler


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
        for char in input_str:
            if char.isalnum():
                count += 1

        # return -1 when the string only contains 1 special character
        if count == 0:
            count = -1
        return count


# train logs contains about 5000 logs without labels
train_logs = pd.read_csv("./dataset/train_logs.csv")
train_scores = pd.read_csv("./dataset/train_scores.csv")
test_logs = pd.read_csv("./dataset/test_logs.csv")

# one hot encode the activity attribute 
categories_in_activity = ["Nonproduction", "Input", "Remove/Cut", "Replace", "Paste"]
train_logs.loc[~train_logs["activity"].isin(categories_in_activity), "activity"] = "Move"
one_hot_encoder = pd.get_dummies(train_logs["activity"], dtype=int)
train_logs = train_logs.drop(columns=["activity"])
train_logs = pd.concat([train_logs, one_hot_encoder], axis=1)

train_logs["text_change"] = train_logs["text_change"].apply(count_char)

# remove categorical attributes
train_logs = train_logs.drop(columns=["down_event", "up_event"])

time_gap = []
gap_pct = []

for _, group in train_logs.groupby("id"):
    diff = group["down_time"] - group["up_time"].shift(1).fillna(0)
    time_gap.append(diff)
    gap_pct.append(diff / diff.sum())

pause = pd.concat(time_gap, axis=0)
pause.name = "pause"
pause_pct = pd.concat(gap_pct, axis=0)
pause_pct.name = "pause_pct"

train_logs = pd.concat([train_logs, pause, pause_pct], axis=1)

# no_scale_columns = train_logs[['id','text_change']]
# train_logs = train_logs.drop(columns=['id','text_change'])
# scaler = StandardScaler()
# train_logs = pd.DataFrame(scaler.fit_transform(train_logs), columns=train_logs.columns)

# train_logs = pd.concat([train_logs, one_hot_encoder, no_scale_columns], axis=1)

# set index using id
train_logs = train_logs.set_index(["id"])
train_scores = train_scores.set_index(["id"])

train_logs.to_csv("./dataset/new_train_logs.csv")
train_scores.to_csv("./dataset/new_train_scores.csv")

'''
feature engineering for each essay
'''
index = []
mean_pause = []
del_sum = []
add_sum = []
del_len = []
add_len = []

for idx, group in train_logs.groupby("id"):
    index.append(idx)

    mean_pause.append(group["pause"].mean())

    del_sum.append((group["Remove/Cut"] == 1).sum())

    add_sum.append((group["Input"] == 1).sum() + (group["Paste"] == 1).sum())

    diff = group["word_count"].diff()
    neg = diff[diff.fillna(0) < 0]
    pos= diff[diff.fillna(0) > 0]
    del_len.append(-neg.sum())
    add_len.append(pos.sum())

train_essays = pd.DataFrame({"mean_pause": mean_pause, 
                             "num_del": del_sum,
                             "num_insert": add_sum, 
                             "words_del": del_len, 
                             "words_insert": add_len}, index=index
                             )

train_essays.to_csv("./dataset/train_essays.csv")