# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 15:17:02 2021

@author: cjesus
"""

''' Initialize Environment '''

# import packages
import pandas as pd
import numpy as np

# import machine learning model
from sklearn.tree import DecisionTreeClassifier

# import a data splitter to make a testing and training dataset
from sklearn.model_selection import train_test_split

# import accuracy score for assessing a model after testing it
from sklearn.metrics import accuracy_score

# change pandas to display more stuff (it's my preference)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 300)






































#%%

''' load original data in once '''
data = pd.read_csv(r"C:\Users\cjesus\Downloads\archive(2)\aug_train.csv")

#%%

# test that the data import worked
print(data.head())

#%%

''' Clean The Data '''

# Drop "useless" columns

data = data[["enrolled_university", "gender", "education_level",
             "target"]]
# Drop rows without a value in columns "gender" and "education level" and
# "enrolled university"
print(len(data.gender))
data = data.dropna(subset=["enrolled_university", "gender", "education_level"])
print(len(data.gender))

# one-hot encode
onehot = pd.get_dummies(data['education_level'])
data = data.join(onehot)

wonhot = pd.get_dummies(data['enrolled_university'])
data = data.join(wonhot)

juanhot = pd.get_dummies(data['gender'])
data = data.join(juanhot)

# get rid of useless columns
data = data.drop("education_level", axis=1)
data = data.drop("enrolled_university", axis=1)
data = data.drop("gender", axis=1)

#%%

print(data.head())
#%%

# initialize the classifier
knn = KNeighborsClassifier(n_neighbors=3)

# create a dataset for training and one for testing: 80% train, 20% test
train, test = train_test_split(data, test_size = 0.2)

# separate each dataset for features and the target
train_features = train.drop("target", axis=1)
test_features = test.drop("target", axis=1)

train_labels = train["target"]
test_labels = test["target"]

#%%
# train your model!!!
knn = knn.fit(train_features, train_labels)
# generate possible targets
predict = knn.predict(test_features)

print(accuracy_score(test_labels, predict))