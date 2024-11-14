# code for svm classifier

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
dfPredict = dfPredict.drop(["id"], axis=1)

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

# use one-hot encoder to convert each catagorical variable to T/F format
def oneHotBind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[feature_to_encode])
    result = pd.concat([original_dataframe, dummies], axis=1)
    result = result.drop(feature_to_encode, axis=1)
    return result

dfTrain = oneHotBind(dfTrain, ["date_account_created", "timestamp_first_active", "date_first_booking", "gender", "signup_method", "signup_flow", "language", "affiliate_channel", "affiliate_provider", "first_affiliate_tracked", "signup_app", "first_device_type", "first_browser"])
dfTest = oneHotBind(dfTest, ["date_account_created", "timestamp_first_active", "date_first_booking", "gender", "signup_method", "signup_flow", "language", "affiliate_channel", "affiliate_provider", "first_affiliate_tracked", "signup_app", "first_device_type", "first_browser"])
dfPredict = oneHotBind(dfPredict, ["date_account_created", "timestamp_first_active", "date_first_booking", "gender", "signup_method", "signup_flow", "language", "affiliate_channel", "affiliate_provider", "first_affiliate_tracked", "signup_app", "first_device_type", "first_browser"])

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

# convert destination country to binary
from sklearn import preprocessing

def encode_country(dataset):
    le = preprocessing.LabelEncoder()
    le = le.fit(dataset['country_destination'])
    dataset['country_destination'] = le.transform(dataset['country_destination'])
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    return dataset, mapping

dfTrain, mapping = encode_country(dfTrain)
dfTest, _ = encode_country(dfTest)

dfTrain = dfTrain.head(int(len(dfTrain.index) / 10))
dfTest = dfTest.head(int(len(dfTest.index) / 10))

# seperate X and Y (tuple and class)
X_train = dfTrain.loc[:,dfTrain.columns !='country_destination'].values
Y_train = dfTrain['country_destination'].values
X_test = dfTest.loc[:,dfTest.columns !='country_destination'].values
Y_test = dfTest['country_destination'].values

# svm

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from spinner import Spinner

# train and test model

print("Training model...")

# display spinner while model is being trained and tested
with Spinner():
    svm = SVC(gamma='auto')
    svm.fit(X_train, Y_train)

print("Testing model...")

with Spinner():
    predictions = svm.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    report = classification_report(Y_test, predictions, zero_division=1)

print("=======================================================")
print("SVM model:")
print("Accuracy: " + str(accuracy_score(Y_test, predictions)))
print(report)

print("Making predictions...")

# display spinner while predictions are being made
with Spinner():
    predictions = svm.predict(dfPredict.iloc[:,1:].values)

# decode country
decoded_predictions = ["nil"] * len(predictions)
for i in range(len(predictions)):
    for key, value in mapping.items():
        if predictions[i] == value:
            decoded_predictions[i] = key
        
frame = {
    "id": ids,
    "country": decoded_predictions
    }

output = pd.DataFrame(frame)

predictions_file = "../predictions/svm_predictions.csv"

output.to_csv(predictions_file, index=False)

print(f"Wrote predictions to {predictions_file}")

