import pathlib

import pandas as pd

from utils import count_char

BASE_PATH = "./linking-writing-processes-to-writing-quality"
OUT_PATH = "./data/preprocessed"

ACTIVITY_CATEGORIES = [
    "Nonproduction", "Input", "Remove/Cut", "Replace", "Paste"
]

def preprocess_logs(filepath, train_out_dir, test_out_dir, cutoff=None, train_pct=0.8,
                    one_hot_activities=True):
    """
    Preprocess the big csv with all the logs in it:
    - splits into individual csvs for each essay
    - converts the activity into either enums or a one-hot encoding
    - drops the down_time and up_time columns
    - drops the down_event and up_event columns
    - drops the event_id column
    - converts the text_change into a numerical count
    """

    # create the out directories if they don't exist
    pathlib.Path(train_out_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(test_out_dir).mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(filepath)
    ids = data["id"].unique().tolist()

    # either convert the activities to individual one-hot (0 || 1) 
    # columns or a single column with enum values
    data.loc[~data["activity"].isin(ACTIVITY_CATEGORIES), "activity"] = "Move"
    if one_hot_activities:
        one_hot_encoder = pd.get_dummies(data["activity"], dtype=int)
        data = data.drop(columns=["activity"])
        data = pd.concat([data, one_hot_encoder], axis=1)
    else:
        data["activity"] = data["activity"].apply(
            lambda a: (ACTIVITY_CATEGORIES + ["Move"]).index(a)
        )

    # drop columns we won't be using
    data = data.drop(columns=["down_time", "up_time"])
    data = data.drop(columns=["down_event", "up_event"])
    data = data.drop(columns=["event_id"])
    
    # default to preprocessing all essays
    if cutoff is None:
        cutoff = len(ids)

    for idx, id in enumerate(ids):
        out_dir = train_out_dir

        # optionally only split out the first n
        if idx >= cutoff:
            break

        # implement the train / test split
        if (idx / cutoff) > train_pct:
            out_dir = test_out_dir
        
        # find the essay, then drop the id column (causes issues with loading
        # into a tensor later)
        essay = data[data["id"] == id]
        essay = essay.drop(columns=["id"])

        # convert text change to a numerical value
        essay["text_change"] = essay["text_change"].apply(count_char)

        essay.to_csv(f"{out_dir}/{id}.csv", index=False)

def main():
    use_one_hot_activities = False
    train_logs_dir = ("train_logs_split_one_hot" if use_one_hot_activities
        else "train_logs_split_enum")
    test_logs_dir = ("test_logs_split_one_hot" if use_one_hot_activities
        else "test_logs_split_enum")

    preprocess_logs(
        f"{BASE_PATH}/train_logs.csv", 
        f"{OUT_PATH}/{train_logs_dir}", 
        f"{OUT_PATH}/{test_logs_dir}", 
        cutoff=500, 
        train_pct=0.8,
        one_hot_activities=use_one_hot_activities
    )

if __name__ == "__main__":
    main()