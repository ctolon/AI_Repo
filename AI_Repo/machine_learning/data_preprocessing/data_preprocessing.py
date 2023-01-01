#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Cevat Batuhan Tolon
"""

import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# =====================================================
# Load Data
# =====================================================
my_missing_data = pd.read_csv("data/missing_datas.csv")
print(my_missing_data.describe())

# =====================================================
# Handle NaN columns with mean value strategy
# =====================================================
my_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

# Before the transformation
age = my_missing_data.iloc[:,1:4].values
print("Age with NaN samples, Before the transformation\n\n", age)

# After the transformation
my_imputer = my_imputer.fit(age[:,1:4])
age[:,1:4] = my_imputer.transform(age[:,1:4])
print("Age without NaN samples, After the transformation\n\n", age)

# print nation column samples
nation = my_missing_data.iloc[:,0:1].values
print("Nation Samples before encoding\n\n", nation)

# =====================================================
# Encode categorical to numeric --> nation column samples as np arrays
# =====================================================
label_encoding = preprocessing.LabelEncoder()
nation[:,0] = label_encoding.fit_transform(my_missing_data.iloc[:,0])
print("Nation Samples after label encoding\n\n", nation)

ohe = preprocessing.OneHotEncoder()
nation = ohe.fit_transform(nation).toarray()
print("Nation Samples After ohe encoding\n\n", nation)

# =====================================================
# Merge DataFrames with columns
# =====================================================

result = pd.DataFrame(data=nation, index = range(22), columns= ['fr','tr', 'us'])
print("Nation Dataframe:\n", result)

result2 = pd.DataFrame(data=age, index = range(22), columns = ['boy','kilo','yas'])
print("Age Dataframe :\n\n", result2)

gender = my_missing_data.iloc[:,-1].values
result3 = pd.DataFrame(data=gender, index=range(22), columns=['cinsiyet'])
print("Gender Dataframe :\n\n", result3)

# =====================================================
# Numpy to DF
# =====================================================
# Merge Nation and Age DataFrames
s = pd.concat([result,result2], axis=1)
print("Merged Nation and Age DF: \n\n",s)

# Merge Nation-Age and Gender DataFrames
s2 = pd.concat([s,result3], axis=1)
print("Merged Nation-Age and Gender DF: \n\n",s2)

# Train-test split for ML
x_train, x_test, y_train, y_test = train_test_split(s, result3, test_size=0.33, random_state=0)

# Create Standart Scaler object and apply
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

print("X_train:\n\n",X_train)
print("X_test:\n\n",X_test)


