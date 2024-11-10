# code for decision tree classifier

import sys

n = 1
if len(sys.argv) > 1:
    try:
        n = int(sys.argv[1])
    except:
        pass


import pandas as pd

# get train and test dataframes
dfTrain = pd.read_csv("../datasets/train_users_2.csv", skipinitialspace=True)
dfTest = pd.read_csv("../datasets/test.csv", skipinitialspace=True)

# remove tuples with unknown, untracked, empty, and NaN values
dfTrain = dfTrain[(dfTrain.values != "-unknown-").all(axis=1)]
dfTrain = dfTrain[(dfTrain.values != "untracked").all(axis=1)]
dfTrain = dfTrain[(dfTrain.values != "").all(axis=1)]
dfTrain = dfTrain[(dfTrain.values != "NaN").all(axis=1)]
dfTrain = dfTrain.dropna()

dfTest = dfTest[(dfTest.values != "-unknown-").all(axis=1)]
dfTest = dfTest[(dfTest.values != "untracked").all(axis=1)]
dfTrain = dfTrain[(dfTrain.values != "").all(axis=1)]
dfTest = dfTest[(dfTest.values != "NaN").all(axis=1)]
dfTest = dfTest.dropna()

# drop id attribute
dfTrain = dfTrain.drop(["id"], axis=1)
dfTest = dfTest.drop(["id"], axis=1)

# for date/timestamp attributes, reduce them to just the year

dfTrain["timestamp_first_active"] = dfTrain["timestamp_first_active"].astype("string").str.slice(stop=4)
dfTest["timestamp_first_active"] = dfTest["timestamp_first_active"].astype("string").str.slice(stop=4)

dfTrain["date_account_created"] = dfTrain["date_account_created"].str.slice(stop=4)
dfTest["date_account_created"] = dfTest["date_account_created"].str.slice(stop=4)

dfTrain["date_first_booking"] = dfTrain["date_first_booking"].str.slice(stop=4)
dfTest["date_first_booking"] = dfTest["date_first_booking"].str.slice(stop=4)


# use one-hop encoder to convert each catagorical variable to T/F format
def oneHotBind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[feature_to_encode])
    result = pd.concat([original_dataframe, dummies], axis=1)
    result = result.drop(feature_to_encode, axis=1)
    return result

dfTrain = oneHotBind(dfTrain, ["date_account_created", "timestamp_first_active", "date_first_booking", "gender", "age", "signup_method", "signup_flow", "language", "affiliate_channel", "affiliate_provider", "first_affiliate_tracked", "signup_app", "first_device_type", "first_browser"])
dfTest = oneHotBind(dfTest, ["date_account_created", "timestamp_first_active", "date_first_booking", "gender", "age", "signup_method", "signup_flow", "language", "affiliate_channel", "affiliate_provider", "first_affiliate_tracked", "signup_app", "first_device_type", "first_browser"])

# add missing attributes (make sure test set contains training set attributes)
for attribute in dfTrain.keys():
    if attribute not in dfTest.keys():
        print(f"Adding missing feature {attribute}")
        list = [False] * len(dfTest.index)
        #newDf = pd.DataFrame({attribute: list})
        #dfTest = pd.concat([dfTest, newDf], axis=1)
        dfTest[attribute] = False

# seperate X and Y (tuple and class)

X_train, Y_train = dfTrain.iloc[:,1:].values, dfTrain.iloc[:, 0].values
X_test, Y_test = dfTest.iloc[:,1:].values, dfTest.iloc[:, 0].values

# decision tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from tqdm import tqdm

accuracy_scores = []

# compute mean accuracy
for _ in tqdm(range(n), desc=f"Computing mean accuracy of {n} models"):
    d_tree = DecisionTreeClassifier()
    d_tree.fit(X_train, Y_train)
    predictions = d_tree.predict(X_test)
    accuracy_scores.append(accuracy_score(Y_test, predictions))


import statistics

print("=======================================================")
print("Decison Tree Model:")
print("Mean Accuracy: " + str(statistics.mean(accuracy_scores)))
print(classification_report(Y_test, predictions, zero_division=1))
