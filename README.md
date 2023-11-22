# CSCI5527-course-project

Dataset available on https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/data

## Notes from Sam on splitting into individual csvs via `preprocess.py`

This can be run to split the large csv file into individual ones for each essay - it only needs to be
run once. This is useful for:
- Testing out model architectures on a subset of the essays, by using the `cutoff` parameter to preprocess only a subset
and use that as the model inputs.
- Training models without reading the big csv into memory at once (helps a lot with speed on Sam's computer).

Preprocessing splits the dataset into training and test based on the `train_pct` parameter.

Preprocessing doesn't automatically preprocess the `train_scores.csv` csv file since it isn't that big. Instead, the 
corresponding `SplitDataset` class reads it in and uses the id to match the score to an essay. I recommend `cp`-ing the
`train_scores.csv` file to `test_scores.csv` or just using the same scores file for both stages
since extra scores will be ignored by the `SplitDataset` class anyway.

See `train_one_hot_conv_model.py` and `train_enum_conv_model.py` for examples.