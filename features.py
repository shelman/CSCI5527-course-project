import pandas as pd
import numpy as np
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
        if input_str == " ":
            return count
        
        for char in input_str:
            if char == "=":
                count = 0

            if char.isalnum():
                count += 1
        return count



train_logs = pd.read_csv("./dataset/train_logs.csv") # train logs contains about 5000 logs without labels
train_scores = pd.read_csv("./dataset/train_scores.csv")
# test_logs = pd.read_csv("./dataset/test_logs.csv")

'''
New features: pause
'''
time_gap = []

for _, group in train_logs.groupby("id"):
    diff = group["down_time"] - group["up_time"].shift(1).fillna(0)
    time_gap.append(diff)

pause = pd.concat(time_gap, axis=0)
pause.name = "pause"
pause[pause < 0] = 0


train_logs = pd.concat([train_logs, pause], axis=1)

'''
Process the attribute activity for one hot encoding
Normalize attributes with large values
'''
categories_in_activity = ["Nonproduction", "Input", "Remove/Cut", "Replace", "Paste"]
train_logs.loc[~train_logs["activity"].isin(categories_in_activity), "activity"] = "Move"
one_hot_encoder = pd.get_dummies(train_logs["activity"], dtype=int)

# normalization
# no_scale_columns = train_logs[["id", "text_change", "cursor_position"]]
# train_logs = train_logs.drop(columns=["activity", "event_id", "id", 
#                                       "down_event", "up_event", 
#                                       "cursor_position", "text_change"])
# scaler = StandardScaler()
# train_logs = pd.DataFrame(scaler.fit_transform(train_logs), columns=train_logs.columns)
# train_logs = pd.concat([train_logs, one_hot_encoder, no_scale_columns], axis=1)

train_logs = pd.concat([train_logs, one_hot_encoder], axis=1)

'''
Remove categorical attributes
'''
train_logs = train_logs.drop(columns=["activity", "event_id", "down_event", "up_event"])
# new_test_logs = test_logs.drop(columns=["event_id", "down_event", "up_event"])

'''
Replace the attribute text_change with numerical values
'''
train_logs["text_change"] = train_logs["text_change"].apply(count_char)
# test_logs["text_change"] = test_logs["text_change"].apply(count_char)


'''
set index using id
'''
train_logs = train_logs.set_index(["id"])
train_scores = train_scores.set_index(["id"])
# test_logs = test_logs.set_index(["id"])

train_logs.to_csv("./dataset/new_train_logs.csv")
train_scores.to_csv("./dataset/new_train_scores.csv")
# new_test_logs.to_csv("./dataset/new_test_logs.csv")



'''
feature engineering for each essay
'''
index = []
pause_num = []
mean_pause = []
del_num = []
add_num = []
del_len = []
add_len = []
del_ratio = []
add_ratio = []
product_process_ratio = []
p_bursts = []
r_bursts = []
p_bursts_num = []
r_bursts_num = []

for idx, group in train_logs.groupby("id"):
    index.append(idx)

    pause_num.append((group["pause"] > 0).sum())

    mean_pause.append(group["pause"].mean())

    remove = group["Remove/Cut"] == 1
    input = group["Input"] == 1
    # number of deletions
    del_num.append(remove.sum())

    # number of insertions
    add_num.append(input.sum() + (group["Paste"] == 1).sum())

    # length of deletions & length of insertions
    remove_char_sum = group.loc[remove, "text_change"].sum()
    add_char_sum = group.loc[input, "text_change"].sum()
    del_len.append(remove_char_sum)
    add_len.append(add_char_sum)

    # proportion of deletions & proportion of insertions
    # (as a % of total writing time)
    total_time = group.iloc[-1]["up_time"]
    del_time = group.loc[remove, "action_time"].sum()
    add_time = group.loc[input, "action_time"].sum()
    del_ratio.append(del_time)
    add_ratio.append(add_time)

    # product vs process ratio (The number of characters 
    # in the product divided by the number of characters 
    # produced during the writing process)
    total_char = add_char_sum
    result_char = add_char_sum - remove_char_sum
    product_process_ratio.append(result_char / total_char)

    # number of P-bursts & number of R-bursts
    p_counter = 0
    r_counter = 0
    p_char = 0
    r_char = 0
    word_count = group["word_count"].shift(1)
    for i, g in group.iterrows():
        if g["pause"] >= 1000 and g["Input"] == 1:
            p_counter += 1
            r_char = 0
            continue
        elif g["Input"] == 1:
            p_char += g["text_change"]

        if g["Nonproduction"] == 1 or g["Paste"] == 1 or g["Replace"] == 1:
            r_counter += 1
            p_char = 0
            continue
        elif g["pause"] >= 1000 and g["Remove/Cut"] == 1:
            r_counter +=1
            p_char = 0
            continue
        elif g["Input"] == 1:
            r_char += g["text_change"]
        
    p_bursts.append(p_counter)
    r_bursts.append(r_counter)
    p_bursts_num.append(p_char)
    r_bursts_num.append(r_char)


train_essays = pd.DataFrame({"pause_num": pause_num,
                             "mean_pause": mean_pause, 
                             "del_num": del_num,
                             "insert_num": add_num, 
                             "del_length": del_len, 
                             "insert_length": add_len, 
                             "del_ratio": del_ratio, 
                             "insert_ratio": add_ratio, 
                             "product_process_ratio": product_process_ratio,
                             "p_bursts": p_bursts,
                             "r_bursts": r_bursts,
                             "p_bursts_num": p_bursts_num,
                             "r_bursts_num": r_bursts_num
                             }, index=index
                             )

train_essays.to_csv("./dataset/train_essays.csv")