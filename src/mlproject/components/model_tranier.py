import os
import sys
from dataclasses import dataclass
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
from src.mlproject.utils import evaluate_models,save_object
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import warnings
warnings.filterwarnings("ignore")



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
        
    def eval_metrics(self,actual, pred):
        #f_score = f1_score(actual,pred)
        #roc_auc = roc_auc_score(actual, pred)
        accuracy = accuracy_score(actual, pred)
        return accuracy    
        
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splitting data into train and test")
            
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models= {
                "Logistic":LogisticRegression(),
                "DecisionTree": DecisionTreeClassifier(),
                "RandomForest": RandomForestClassifier(),
                "GradientBoost": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Catboost": CatBoostClassifier(verbose=False),
            }
            
            params = {
    "Logistic": {
        "penalty": ["l1", "l2"],
        "C": [0.001, 0.01, 0.1, 1, 10],
    },
    "DecisionTree": {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "RandomForest": {
        "n_estimators": [50, 100, 200],
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "GradientBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 4, 5,8],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
    },
    "Catboost": {
        "iterations": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "depth": [3, 4, 5],
    },
        }
            
            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models,params)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

             ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            print("This is the best model:")
            print(best_model_name)

            model_names = list(params.keys())

            actual_model=""

            for model in model_names:
                if best_model_name == model:
                    actual_model = actual_model + model

            best_params = params[actual_model]

            mlflow.set_registry_uri("https://dagshub.com/Ayan022000/ML-Project-.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # mlflow

            with mlflow.start_run():

                predicted_qualities = best_model.predict(X_test)

                accuracy = self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_params(best_params)

                #mlflow.log_metric("f_1_score",f_score )
                #mlflow.log_metric("roc_auc_score", roc_auc)
                mlflow.log_metric("accuracy_score", accuracy)


                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
                else:
                    mlflow.sklearn.log_model(best_model, "model")


            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            Accuracy_score = accuracy_score(y_test, predicted)
            return Accuracy_score
            
            
        except Exception as e:
            raise CustomException(e,sys)        