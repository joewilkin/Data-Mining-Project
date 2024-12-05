# code for neural network classifier

import pandas as pd
import numpy as np

# get train and test dataframes
dfTrain = pd.read_csv("../datasets/train_users_2.csv", skipinitialspace=True)


# get tuples whose classes need to be predicted
dfPredict = pd.read_csv("../datasets/test_users.csv", skipinitialspace=True)

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


# Convert to binary value for numerical attributed based on their mean value
def numericalBinary(dataset, features):
    dataset[features] = np.where(dataset[features] >= dataset[features].mean(), 1, 0)

numericalBinary(dfTrain, ['age'])
numericalBinary(dfPredict, ['age'])

# use one-hot encoder to convert each catagorical variable to T/F format
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

# convert destination country to binary
from sklearn import preprocessing

def encode_country(dataset):
    le = preprocessing.LabelEncoder()
    le = le.fit(dataset['country_destination'])
    dataset['country_destination'] = le.transform(dataset['country_destination'])
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    return dataset, mapping

dfTrain, mapping = encode_country(dfTrain)

#dfTrain = dfTrain.head(int(len(dfTrain.index) / 10))
#dfTest = dfTest.head(int(len(dfTest.index) / 10))

from sklearn.model_selection import train_test_split

# seperate X and Y (tuple and class)
X = dfTrain.loc[:,dfTrain.columns !='country_destination'].values
Y = dfTrain['country_destination'].values

# get train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


# neural network

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from spinner import Spinner

# train and test model

print("Training model...")

# display spinner while model is being trained and tested
with Spinner():
    mlp = MLPClassifier(hidden_layer_sizes=(104,104,104), max_iter=1000)
    mlp.fit(X_train,Y_train)

print("Testing model...")

with Spinner():
    predictions = mlp.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    report = classification_report(Y_test, predictions, zero_division=1)

print("=======================================================")
print("Neural Network:")
print("Accuracy: " + str(accuracy_score(Y_test, predictions)))
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
    predictions = mlp.predict(dfPredict.iloc[:,1:].values)
    class_probabilities = mlp.predict_proba(dfPredict.iloc[:,1:].values)

    expanded_ids = []
    expanded_countries = []

    num = 0
    for obs in class_probabilities:
        probs = []
        for i in range(len(obs)):
            if obs[i] > 0:
                probs.append([mlp.classes_[i], obs[i]])
        probs = bubble_sort(probs)
        for p in probs:
            expanded_ids.append(ids[num])
            expanded_countries.append(p[0])
        num += 1

"""
# decode country
decoded_predictions = ["nil"] * len(predictions)
for i in range(len(predictions)):
    for key, value in mapping.items():
        if predictions[i] == value:
            decoded_predictions[i] = key
"""

# decode country
decoded_expanded_countries = ["nil"] * len(expanded_countries)
for i in range(len(expanded_countries)):
    for key, value in mapping.items():
        if expanded_countries[i] == value:
            decoded_expanded_countries[i] = key
"""    
frame = {
    "id": ids,
    "country": decoded_predictions
    }
"""
frame = {
    "id": expanded_ids,
    "country": decoded_expanded_countries
    }

output = pd.DataFrame(frame)

predictions_file = "../predictions/neural_network_predictions.csv"

output.to_csv(predictions_file, index=False)

print(f"Wrote predictions to {predictions_file}")

