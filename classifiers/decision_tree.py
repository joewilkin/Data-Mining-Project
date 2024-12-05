# code for decision tree classifier

import pandas as pd
from sklearn.model_selection import train_test_split

# get training dataset
dfTrain = pd.read_csv("../datasets/train_users_2.csv", skipinitialspace=True)

# get tuples whose classes need to be predicted
dfPredict = pd.read_csv("../datasets/test_users.csv", skipinitialspace=True)

# save ids of tuples to be predicted
ids = dfPredict["id"].values

# drop id attribute
dfTrain = dfTrain.drop(["id"], axis=1)
dfPredict = dfPredict.drop(["id"], axis=1)

# drop date attributes
dfTrain = dfTrain.drop(["date_first_booking"], axis=1)
dfPredict = dfPredict.drop(["date_first_booking"], axis=1)

dfTrain = dfTrain.drop(["date_account_created"], axis=1)
dfPredict = dfPredict.drop(["date_account_created"], axis=1)

dfTrain = dfTrain.drop(["timestamp_first_active"], axis=1)
dfPredict = dfPredict.drop(["timestamp_first_active"], axis=1)

# one-hot encoder to convert each catagorical variable to binary T/F format
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

# seperate X and Y (tuple and class)
Y = dfTrain.iloc[:, 1].values
X = pd.concat([dfTrain.iloc[:, :1], dfTrain.iloc[:, 2:]], axis=1).values

# get train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# decision tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from spinner import Spinner

# train and test model

print("Training model...")

# display spinner while model is being trained and tested
with Spinner():
    d_tree = DecisionTreeClassifier()
    d_tree.fit(X_train, Y_train)

print("Testing model...")

with Spinner():
    predictions = d_tree.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    report = classification_report(Y_test, predictions, zero_division=1)

print("=======================================================")
print("Decison Tree Model:")
print("Accuracy:", accuracy)
print(report)

print("Making predictions...")

# bubble sort to sort countries by probability
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
    predictions = d_tree.predict(dfPredict.iloc[:,1:].values)
    # get probability of each country
    class_probabilities = d_tree.predict_proba(dfPredict.iloc[:,1:].values)
    
    expanded_ids = []
    expanded_countries = []

    # for each observation, get five most probable destination countries
    num = 0
    for obs in class_probabilities:
        probs = []
        for i in range(len(obs)):
            if obs[i] > 0:
                probs.append([d_tree.classes_[i], obs[i]])
        probs = bubble_sort(probs)
        for p in probs:
            expanded_ids.append(ids[num])
            expanded_countries.append(p[0])
        num += 1

# make prediction dataframe
frame = {
    "id": expanded_ids,
    "country": expanded_countries
    }
output = pd.DataFrame(frame)

# write precitions to file
predictions_file = "../predictions/decision_tree_predictions.csv"
output.to_csv(predictions_file, index=False)
print(f"Wrote predictions to {predictions_file}")

# plot tree

import matplotlib.pyplot as plt
from sklearn import tree

tree.plot_tree(d_tree, max_depth=4, filled=True, feature_names=dfTrain.columns, class_names=d_tree.classes_)
plt.show()



