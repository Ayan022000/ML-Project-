import sys
from dataclasses import dataclass
import os
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
from src.mlproject.utils import save_object



@dataclass
class DataTransfornationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config =DataTransfornationConfig()
        
    def get_data_transformer_object(self):
        try:
            categorical_columns = [' education', ' self_employed']
            numerical_columns = [' no_of_dependents',
                                ' income_annum',
                                ' loan_amount',
                                ' loan_term',
                                ' cibil_score',
                                ' residential_assets_value',
                                ' commercial_assets_value',
                                ' luxury_assets_value',
                                ' bank_asset_value']
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('labelencoder',OneHotEncoder())
                ]
            )
            
            logging.info("Numerical columns tranformation completed")
            logging.info("Categorical column tranformation completed")
            
            
            preprocessor = ColumnTransformer(
                [
                    ('num_pipelines',num_pipeline,numerical_columns),
                    ('cat_pipelines',cat_pipeline,categorical_columns)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name=" loan_status"
            categorical_columns = [' education', ' self_employed']
            

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)