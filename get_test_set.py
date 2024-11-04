# get the test set from the training set

import pandas as pd

# import training set
dfTrain = pd.read_csv("./datasets/train_users_2.csv", skipinitialspace=True)

# initalize empty test set dataframe
dfTest = pd.DataFrame(columns=dfTrain.keys())

# get length of test set
length = int(len(dfTrain.index) / 3)

from random import randrange
from tqdm import tqdm

# randomly select a row from the training set and move it to the test set
# display the loading bar
for _ in tqdm (range (length), desc="Loading..."):
    i = randrange(0, len(dfTrain.index))
    dfTest.loc[len(dfTest.index)] = dfTrain.iloc[i].values
    dfTrain = dfTrain.drop(dfTrain.index[i])

print(dfTrain)
print(dfTest)

# export test set as csv file
dfTest.to_csv("./datasets/test.csv", index=False)


