# code for random forest classifier

import pandas as pd
from sklearn.model_selection import train_test_split

# get train dataframe
dfTrain = pd.read_csv("../datasets/train_users_2.csv", skipinitialspace=True)

# get tuples whose classes need to be predicted
dfPredict = pd.read_csv("../datasets/test_users.csv", skipinitialspace=True)

"""
# remove tuples with unknown, untracked, empty, and NaN values
dfTrain = dfTrain[(dfTrain.values != "-unknown-").all(axis=1)]
dfTrain = dfTrain[(dfTrain.values != "untracked").all(axis=1)]
dfTrain = dfTrain[(dfTrain.values != "").all(axis=1)]
dfTrain = dfTrain[(dfTrain.values != "NaN").all(axis=1)]
dfTrain = dfTrain.dropna()
"""

# save ids of tuples to be predicted
ids = dfPredict["id"].values

# drop id attribute
dfTrain = dfTrain.drop(["id"], axis=1)
dfPredict = dfPredict.drop(["id"], axis=1)

# for date/timestamp attributes, reduce them to just the year
dfTrain["timestamp_first_active"] = dfTrain["timestamp_first_active"].astype("string").str.slice(stop=4)
dfPredict["timestamp_first_active"] = dfPredict["timestamp_first_active"].astype("string").str.slice(stop=4)

dfTrain["date_account_created"] = dfTrain["date_account_created"].str.slice(stop=4)
dfPredict["date_account_created"] = dfPredict["date_account_created"].str.slice(stop=4)

dfTrain["date_first_booking"] = dfTrain["date_first_booking"].str.slice(stop=4)
dfPredict["date_first_booking"] = dfPredict["date_first_booking"].astype("string").str.slice(stop=4)


# one-hot encoder to convert each catagorical variable to T/F format
def oneHotBind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[feature_to_encode])
    result = pd.concat([original_dataframe, dummies], axis=1)
    result = result.drop(feature_to_encode, axis=1)
    return result

dfTrain = oneHotBind(dfTrain, ["date_account_created", "timestamp_first_active", "date_first_booking", "gender", "age", "signup_method", "signup_flow", "language", "affiliate_channel", "affiliate_provider", "first_affiliate_tracked", "signup_app", "first_device_type", "first_browser"])
dfPredict = oneHotBind(dfPredict, ["date_account_created", "timestamp_first_active", "date_first_booking", "gender", "age", "signup_method", "signup_flow", "language", "affiliate_channel", "affiliate_provider", "first_affiliate_tracked", "signup_app", "first_device_type", "first_browser"])


# add missing attributes
for attribute in dfTrain.keys():
    if attribute not in dfPredict.keys():
        print(f"Adding missing feature {attribute} to prediction set")
        list = [False] * len(dfPredict.index)
        dfPredict[attribute] = False
        
for attribute in dfPredict.keys():
    if attribute not in dfTrain.keys():
        print(f"Adding missing feature {attribute} to training set")
        list = [False] * len(dfTrain.index)
        dfTrain[attribute] = False

# seperate X and Y (tuple and class)
X, Y = dfTrain.iloc[:,1:].values, dfTrain.iloc[:, 0].values

# get train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# random forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from spinner import Spinner

# train and test model

print("Training model...")

# display spinner while model is being trained and tested
with Spinner():
    forest = RandomForestClassifier()
    forest.fit(X_train, Y_train)

print("Testing model...")

with Spinner():
    predictions = forest.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    report = classification_report(Y_test, predictions, zero_division=1)

print("=======================================================")
print("Random Forest:")
print("Accuracy:", accuracy)
print(report)

print("Making predicions...")

def bubble_sort(list):
    for i in range(len(list) - 1):
        for j in range(len(list) - i - 1):
            if list[j][1] < list[j + 1][1]:
                temp = list[j]
                list[j] = list[j + 1]
                list[j + 1] = temp
    return list

# display spinner while predictions are being made
with Spinner():
    predictions = forest.predict(dfPredict.iloc[:,1:].values)
    class_probabilities = forest.predict_proba(dfPredict.iloc[:,1:].values)

    expanded_ids = []
    expanded_countries = []

    num = 0
    for obs in class_probabilities:
        probs = []
        for i in range(len(obs)):
            if obs[i] > .05 and len(probs) < 5:
                probs.append([forest.classes_[i], obs[i]])
        probs = bubble_sort(probs)
        for p in probs:
            expanded_ids.append(ids[num])
            expanded_countries.append(p[0])
        num += 1

"""
frame = {
    "id": ids,
    "country": predictions
    }
"""

frame = {
    "id": expanded_ids,
    "country": expanded_countries
    }


output = pd.DataFrame(frame)

predictions_file = "../predictions/random_forest_predictions.csv"

output.to_csv(predictions_file, index=False)

print(f"Wrote predictions to {predictions_file}")

