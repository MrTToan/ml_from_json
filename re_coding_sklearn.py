#!/usr/bin/env python
# coding: utf-8

# <b>Problem:</b> <br/>
# 
# Read from a json file which includes params to create models for fraud prediction of Tiki orders. <br/>
# File format should be like this:
# 
# [
# {
#     'model': RandomForestClassifier,
#     'n_estimators': [10, 15],
#     'max_depth': [10, 12] 
# },
# {
#     'model': LinearRegression,
#     'n_jobs': [3, 5]
# }
# ]
# 

# <b>How to do:</b> <br/>
# 1. read text file -> string to json
# 2. build a class of prediction model which use above params
# 3. Loop through list and print result then pass them to a pandas frame
# 4. Compare result 

import json
import ast 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

import warnings
warnings.simplefilter("ignore")

def parse_file_to_config():
    """This function is to parse file into a list of needed params and store them in a list"""
    with open('configs.json', 'r') as f:
        _list = ast.literal_eval(f.read())
        return(_list)

class BaseModel:
    random_state = 16
    def __init__(self, df, target_variable, test_size):
        self.df = df  
        X = self.df.drop(columns=[target_variable])
        y = self.df[target_variable]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size= test_size, random_state=self.random_state)
    
    def _initialize(self, model):
        self.model = model()
    
    def _fit(self):
        self.model.fit(self.X_train, self.y_train)

    def _result(self):
        y_pred_test = self.model.predict(self.X_test)
        return(accuracy_score(self.y_test, y_pred_test))

    def _predict(self, new_obs):
        return(self.model.predict(new_obs))

    def gsc(self):
        model_param = parse_file_to_config()
        result_table = {}
        for models in model_param:
            each_mod = list(models.values())
            gsc_obj = GridSearchCV(eval(each_mod[0])(), param_grid = each_mod[1], scoring="accuracy")
            gsc_obj.fit(self.X_test, self.y_test)
            result_table[each_mod[0]] = gsc_obj.best_score_

        result_df = pd.DataFrame([result_table])
        print(result_df)

# import & process raw data
raw_df = pd.read_csv('order_label4.csv')
raw_df_drop_columns = raw_df.drop(columns=['fraud2', 'order_code'])
# raw_df_drop_columns[:, 1] = LabelEncoder().fit_transform(raw_df_drop_columns[:, 1]) lỗi, đéo hiểu tại sao, sof bảo là cần convert sang numpy array
raw_df_drop_columns["payment_method"] = LabelEncoder().fit_transform(raw_df_drop_columns["payment_method"])
train_df = raw_df_drop_columns 

mymod = BaseModel(train_df, 'fraud', 0.8)
# mymod._initialize(RandomForestClassifier)
# mymod._fit()   
mymod.gsc()