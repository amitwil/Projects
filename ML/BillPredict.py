import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np 

try:
        data = pd.read_csv(r'Data/Billdata.csv')
except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

columns_to_drop = ['Month']
data = data.drop(columns=columns_to_drop)

train_set,test_set=train_test_split(data,test_size=0.2,random_state=42)

# Identify numerical columns
numerical_columns = data.select_dtypes(include=['number']).columns
print("Numerical columns:", numerical_columns)

# Identify non-numerical columns
non_numerical_columns = data.select_dtypes(exclude=['number']).columns
print("Non-numerical columns:", non_numerical_columns)

target_column_name = 'Totalcharges'

if target_column_name in numerical_columns:
    numerical_columns = numerical_columns.drop(target_column_name)
if target_column_name in non_numerical_columns:
     non_numerical_columns = non_numerical_columns.drop(target_column_name)

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from catboost import CatBoostRegressor
num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,non_numerical_columns)

                ]


            )

target_column_name='Totalcharges'
input_feature_train=train_set.drop(columns=[target_column_name],axis=1)
target_feature_train=train_set[target_column_name]

input_feature_test=test_set.drop(columns=[target_column_name],axis=1)
target_feature_test=test_set[target_column_name]

input_feature_train_arr=preprocessor.fit_transform(input_feature_train)
input_feature_test_arr=preprocessor.transform(input_feature_test)

train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train)
            ]
test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test)]

X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

print('X_train',X_train.shape)
print('y_train',y_train.shape)
print('X_test',X_test.shape)
print('y_test',y_test.shape)

# X_train,y_train,X_test,y_test=(
#                 input_feature_train_arr[:,:-1],
#                 input_feature_train_arr[:,-1],
#                 input_feature_test_arr[:,:-1],
#                 input_feature_test_arr[:,-1]
#             )

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

models = {
                 "Random Forest": RandomForestRegressor(),
                 "Decision Tree": DecisionTreeRegressor(),
                 "Gradient Boosting": GradientBoostingRegressor(),
                 "Linear Regression": LinearRegression(),
                 "XGBRegressor": XGBRegressor(),
                 "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                 "AdaBoost Regressor": AdaBoostRegressor(),
            }
params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


def evaluate_models(X_train, y_train,X_test,y_test,models,param):
        report = {}
    
        for i in range(len(list(models))):
            try:
                model = list(models.values())[i]
                para=param[list(models.keys())[i]]
    
                gs = GridSearchCV(model,para,cv=3)
                gs.fit(X_train,y_train)
    
                model.set_params(**gs.best_params_)
                model.fit(X_train,y_train)
    
                #model.fit(X_train, y_train)  # Train model
                    
                y_train_pred = model.predict(X_train)
    
                y_test_pred = model.predict(X_test)
    
                train_model_score = r2_score(y_train, y_train_pred)
    
                test_model_score = r2_score(y_test, y_test_pred)
    
                report[list(models.keys())[i]] = test_model_score
            except ValueError as e:
                # Handle specific error types
                if "negative" in str(e):  # Check if the error is related to negative values
                    report[model_name] = {
                        'error': 'ValueError: Some value(s) of y are negative.'
                    }
                else:
                    report[model_name] = {
                        'error': str(e)
                    }

        return report



model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

## To get best model score from dict
best_model_score = max(sorted(model_report.values()))
best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

best_model = models[best_model_name]
predicted=best_model.predict(X_test)
print('best_model',best_model)

r2_square = r2_score(y_test, predicted)
print('r2_square',r2_square)

import pickle

# Save the preprocessor and the model
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

# create an iterator object with write permission - model.pkl
with open('Bill_model.pkl', 'wb') as files:
    pickle.dump(best_model, files)

# with open('model_pkl' , 'rb') as f:
#     lr = pickle.load(f)
# print("checvk")
# # check prediction

# lr.predict([[5000]]) # similar
# # return r2_square
