# code for random gradient boosting

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# get train dataframe
dfTrain = pd.read_csv("../datasets/train_users_2.csv", skipinitialspace=True)

# get tuples whose classes need to be predicted
dfPredict = pd.read_csv("../datasets/test_users.csv", skipinitialspace=True)

print("Initial class distribution in country_destination:")
print(dfTrain["country_destination"].value_counts())

# remove tuples with unknown, untracked, empty, and NaN values
"""
dfTrain = dfTrain[(dfTrain.values != "-unknown-").all(axis=1)]
dfTrain = dfTrain[(dfTrain.values != "untracked").all(axis=1)]
dfTrain = dfTrain[(dfTrain.values != "").all(axis=1)]
dfTrain = dfTrain[(dfTrain.values != "NaN").all(axis=1)]
dfTrain = dfTrain.dropna()
"""
critical_columns = [
    "signup_method", "signup_flow", "language", 
    "affiliate_channel", "affiliate_provider", 
    "signup_app", "first_device_type"
]
dfTrain = dfTrain.dropna(subset=critical_columns)

# Handle missing values minimally
dfTrain["gender"] = dfTrain["gender"].replace("-unknown-", "unknown")
dfTrain["age"] = dfTrain["age"].fillna(dfTrain["age"].median())
dfTrain["first_affiliate_tracked"] = dfTrain["first_affiliate_tracked"].replace("untracked", "unknown").fillna("unknown")
dfTrain["first_browser"] = dfTrain["first_browser"].replace("-unknown-", "unknown")

# save ids of tuples to be predicted
ids = dfPredict["id"].values

# drop id attribute
dfTrain = dfTrain.drop(["id"], axis=1)
dfPredict = dfPredict.drop(["id"], axis=1)

# drop date_first_booking bc it does not appear in prediction set
dfTrain = dfTrain.drop(["date_first_booking"], axis=1)
dfPredict = dfPredict.drop(["date_first_booking"], axis=1)

dfTrain = dfTrain.drop(["date_account_created"], axis=1)
dfPredict = dfPredict.drop(["date_account_created"], axis=1)

dfTrain = dfTrain.drop(["timestamp_first_active"], axis=1)
dfPredict = dfPredict.drop(["timestamp_first_active"], axis=1)

print("Class distribution after preprocessing:")
print(dfTrain["country_destination"].value_counts())

# one-hot encoder to convert each catagorical variable to T/F format
def oneHotBind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[feature_to_encode])
    result = pd.concat([original_dataframe, dummies], axis=1)
    result = result.drop(feature_to_encode, axis=1)
    return result

dfTrain = oneHotBind(dfTrain, ["gender", "signup_method", "signup_flow", "language", "affiliate_channel", "affiliate_provider", "first_affiliate_tracked", "signup_app", "first_device_type", "first_browser"])
dfPredict = oneHotBind(dfPredict, ["gender", "signup_method", "signup_flow", "language", "affiliate_channel", "affiliate_provider", "first_affiliate_tracked", "signup_app", "first_device_type", "first_browser"])

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

Y = dfTrain.iloc[:, 1].values
X = pd.concat([dfTrain.iloc[:, :1], dfTrain.iloc[:, 2:]], axis=1).values

# get train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Gradient Boosting Classifier

# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from spinner import Spinner

# train and test model

print("Training model...")

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=17)

# display spinner while model is being trained and tested
with Spinner():
    boost = GradientBoostingClassifier(random_state=17, n_estimators=200, learning_rate=0.05, max_depth=5)
    boost.fit(X_train, Y_train)

print("Testing model...")

predictions = boost.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
report = classification_report(Y_test, predictions, zero_division=1)

print("=======================================================")
print("Gradient Boosting Classifier:")
print("Accuracy:", accuracy)
print(report)

print("Making predictions...")

def bubble_sort(list):
    for i in range(len(list) - 1):
        for j in range(len(list) - i - 1):
            if list[j][1] < list[j + 1][1]:
                temp = list[j]
                list[j] = list[j + 1]
                list[j + 1] = temp
    
    if len(list) > 5:
        return list[:5]
    
    return list

# display spinner while predictions are being made
with Spinner():
    predictions = boost.predict(dfPredict.iloc[:,1:].values)
    class_probabilities = boost.predict_proba(dfPredict.iloc[:,1:].values)

    expanded_ids = []
    expanded_countries = []

    num = 0
    for obs in class_probabilities:
        probs = []
        for i in range(len(obs)):
            if obs[i] > 0:
                probs.append([boost.classes_[i], obs[i]])
        probs = bubble_sort(probs)
        for p in probs:
            expanded_ids.append(ids[num])
            expanded_countries.append(p[0])
        num += 1


frame = {
    "id": expanded_ids,
    "country": expanded_countries
    }


output = pd.DataFrame(frame)

predictions_file = "../predictions/gradient_boost_predictions.csv"

output.to_csv(predictions_file, index=False)

print(f"Wrote predictions to {predictions_file}")


