# code for decision tree classifier

import pandas as pd

# get train and test dataframes
dfTrain = pd.read_csv("./datasets/train_users_2.csv", skipinitialspace=True)
dfTest = pd.read_csv("./datasets/test.csv", skipinitialspace=True)

# remove tuples with unknown, untracked, and NaN values
dfTrain = dfTrain[(dfTrain.values != "-unknown-").all(axis=1)]
dfTrain = dfTrain[(dfTrain.values != "untracked").all(axis=1)]
dfTrain = dfTrain[(dfTrain.values != "NaN").all(axis=1)]

dfTest = dfTest[(dfTest.values != "-unknown-").all(axis=1)]
dfTest = dfTest[(dfTest.values != "untracked").all(axis=1)]
dfTest = dfTest[(dfTest.values != "NaN").all(axis=1)]

# drop a few variables
dfTrain = dfTrain.drop(["id"], axis=1)
dfTest = dfTest.drop(["id"], axis=1)
dfTrain = dfTrain.drop(["timestamp_first_active"], axis=1)
dfTest = dfTest.drop(["timestamp_first_active"], axis=1)
dfTrain = dfTrain.drop(["date_account_created"], axis=1)
dfTest = dfTest.drop(["date_account_created"], axis=1)
dfTrain = dfTrain.drop(["date_first_booking"], axis=1)
dfTest = dfTest.drop(["date_first_booking"], axis=1)

# use one-hop encoder to convert each catagorical variable to binary format
def oneHotBind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[feature_to_encode])
    result = pd.concat([original_dataframe, dummies], axis=1)
    result = result.drop(feature_to_encode, axis=1)
    return result

#dfTrain = oneHotBind(dfTrain, ["id", "date_account_created", "timestamp_first_active", "date_first_booking", "gender", "age", "signup_method", "signup_flow", "language", "affiliate_channel", "affiliate_provider", "first_affiliate_tracked", "signup_app", "first_device_type", "first_browser"])
#dfTest = oneHotBind(dfTest, ["id", "date_account_created", "timestamp_first_active", "date_first_booking", "gender", "age", "signup_method", "signup_flow", "language", "affiliate_channel", "affiliate_provider", "first_affiliate_tracked", "signup_app", "first_device_type", "first_browser"])

dfTrain = oneHotBind(dfTrain, ["gender", "age", "signup_method", "signup_flow", "language", "affiliate_channel", "affiliate_provider", "first_affiliate_tracked", "signup_app", "first_device_type", "first_browser"])
dfTest = oneHotBind(dfTest, ["gender", "age", "signup_method", "signup_flow", "language", "affiliate_channel", "affiliate_provider", "first_affiliate_tracked", "signup_app", "first_device_type", "first_browser"])

# add missing attributes (make sure train and test sets have the same attributes)

for attribute in dfTrain.keys():
    if attribute not in dfTest.keys():
        print(f"Adding missing feature {attribute}")
        list = [False] * len(dfTest.index)
        #newDf = pd.DataFrame({attribute: list})
        #dfTest = pd.concat([dfTest, newDf], axis=1)
        dfTest[attribute] = False

"""
for attribute in dfTest.keys():
    if attribute not in dfTrain.keys():
        #print(f"Adding missing feature {attributes}")
        list = [False] * len(dfTrain.index)
        #newDf = pd.DataFrame({attribute: list})
        #dfTrain = pd.concat([dfTrain, newDf], axis=1)
        dfTrain[attribute] = False
"""

# seperate X and Y (tuple and class)

X_train, Y_train = dfTrain.iloc[:,1:].values, dfTrain.iloc[:, 0].values
X_test, Y_test = dfTest.iloc[:,1:].values, dfTest.iloc[:, 0].values

# decision tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

tree = DecisionTreeClassifier()
tree.fit(X_train, Y_train)
predictions = tree.predict(X_test)

print("=======================================================")
print("Decison Tree Model:")
print("Accuracy: " + str(accuracy_score(Y_test, predictions)))
print(classification_report(Y_test, predictions, zero_division=1))