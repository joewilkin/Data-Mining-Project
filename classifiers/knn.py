# code for knn classifier

import pandas as pd
import numpy as np

# get train and test dataframes
dfTrain = pd.read_csv("../datasets/train_users_2.csv", skipinitialspace=True)
dfTest = pd.read_csv("../datasets/test.csv", skipinitialspace=True)

# get tuples whose classes need to be predicted
dfPredict = pd.read_csv("../datasets/test_users.csv", skipinitialspace=True)

"""
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
"""

# save ids of tuples to be predicted
ids = dfPredict["id"].values

# drop id attribute
dfTrain = dfTrain.drop(["id"], axis=1)
dfTest = dfTest.drop(["id"], axis=1)

# for date/timestamp attributes, reduce them to just the year
dfTrain["timestamp_first_active"] = dfTrain["timestamp_first_active"].astype("string").str.slice(stop=4)
dfTest["timestamp_first_active"] = dfTest["timestamp_first_active"].astype("string").str.slice(stop=4)
dfPredict["timestamp_first_active"] = dfPredict["timestamp_first_active"].astype("string").str.slice(stop=4)

dfTrain["date_account_created"] = dfTrain["date_account_created"].str.slice(stop=4)
dfTest["date_account_created"] = dfTest["date_account_created"].str.slice(stop=4)
dfPredict["date_account_created"] = dfPredict["date_account_created"].str.slice(stop=4)

dfTrain["date_first_booking"] = dfTrain["date_first_booking"].str.slice(stop=4)
dfTest["date_first_booking"] = dfTest["date_first_booking"].str.slice(stop=4)
dfPredict["date_first_booking"] = dfPredict["date_first_booking"].astype("string").str.slice(stop=4)


# Convert to binary value for numerical attributed based on their mean value
def numericalBinary(dataset, features):
    dataset[features] = np.where(dataset[features] >= dataset[features].mean(), 1, 0)

numericalBinary(dfTrain, ['age'])
numericalBinary(dfTest, ['age'])
numericalBinary(dfPredict, ['age'])


# use one-hop encoder to convert each catagorical variable to T/F format
def oneHotBind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[feature_to_encode])
    result = pd.concat([original_dataframe, dummies], axis=1)
    result = result.drop(feature_to_encode, axis=1)
    return result

dfTrain = oneHotBind(dfTrain, ["date_account_created", "timestamp_first_active", "date_first_booking", "gender", "age", "signup_method", "signup_flow", "language", "affiliate_channel", "affiliate_provider", "first_affiliate_tracked", "signup_app", "first_device_type", "first_browser"])
dfTest = oneHotBind(dfTest, ["date_account_created", "timestamp_first_active", "date_first_booking", "gender", "age", "signup_method", "signup_flow", "language", "affiliate_channel", "affiliate_provider", "first_affiliate_tracked", "signup_app", "first_device_type", "first_browser"])
dfPredict = oneHotBind(dfPredict, ["date_account_created", "timestamp_first_active", "date_first_booking", "gender", "age", "signup_method", "signup_flow", "language", "affiliate_channel", "affiliate_provider", "first_affiliate_tracked", "signup_app", "first_device_type", "first_browser"])

# add missing attributes
for attribute in dfTrain.keys():
    if attribute not in dfTest.keys():
        print(f"Adding missing feature {attribute} to test set")
        list = [False] * len(dfTest.index)
        dfTest[attribute] = False
    if attribute not in dfPredict.keys():
        print(f"Adding missing feature {attribute} to prediction set")
        list = [False] * len(dfPredict.index)
        dfPredict[attribute] = False
        
for attribute in dfPredict.keys():
    if attribute not in dfTrain.keys():
        print(f"Adding missing feature {attribute} to training set")
        list = [False] * len(dfTrain.index)
        dfTrain[attribute] = False
    if attribute not in dfTest.keys():
        print(f"Adding missing feature {attribute} to test set")
        list = [False] * len(dfTest.index)
        dfTest[attribute] = False

# seperate X and Y (tuple and class)
X_train, Y_train = dfTrain.iloc[:,1:].values, dfTrain.iloc[:, 0].values
X_test, Y_test = dfTest.iloc[:,1:].values, dfTest.iloc[:, 0].values

# kn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from spinner import Spinner

# train and test model

print("Training model...")

# display spinner while model is being trained and tested
with Spinner():
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, Y_train)

print("Testing model...")

with Spinner():
    predictions = knn.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    report = classification_report(Y_test, predictions, zero_division=1)

print("=======================================================")
print("KNN Model:")
print("Accuracy: " + str(accuracy_score(Y_test, predictions)))
print(report)

print("Making predicions...")

# display spinner while predictions are being made
with Spinner():
    predictions = knn.predict(dfPredict.iloc[:,1:].values)

frame = {
    "id": ids,
    "country": predictions
    }

output = pd.DataFrame(frame)

predictions_file = "../predictions/knn_predictions.csv"

output.to_csv(predictions_file, index=False)

print(f"Wrote predictions to {predictions_file}")
