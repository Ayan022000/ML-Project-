import sys
import pandas as pd
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path= 'artifacts\preprocessor.pkl'
        
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
        
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
        
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
        
    
    
class CustomData:
    def __init__(self,
         no_of_dependents:int,
         education:str,
         self_employed:str,
         income_annum:int,
         loan_amount:int,
         loan_term:int,
         cibil_score:int,
         residential_assets_value:int,
         commercial_assets_value:int,
         luxury_assets_value:int,
         bank_asset_value:int):
    
       self.no_of_dependents= no_of_dependents
       self.education =education
       self.self_employed = self_employed
       self.income_annum = income_annum
       self.loan_amount = loan_amount
       self.loan_term =loan_term
       self.cibil_score = cibil_score
       self.residential_assets_value = residential_assets_value
       self.commercial_assets_value =commercial_assets_value
       self.luxury_assets_value = luxury_assets_value
       self.bank_asset_value = bank_asset_value
       
       
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
    " no_of_dependents": [self.no_of_dependents],
    " education": [self.education],
    " self_employed": [self.self_employed],
    " income_annum": [self.income_annum],
    " loan_amount": [self.loan_amount],
    " loan_term": [self.loan_term],
    " cibil_score": [self.cibil_score],
    " residential_assets_value": [self.residential_assets_value],
    " commercial_assets_value": [self.commercial_assets_value],
    " luxury_assets_value": [self.luxury_assets_value],
    " bank_asset_value": [self.bank_asset_value]
              }
            return pd.DataFrame(custom_data_input_dict)
        
        
        except Exception as e:
            raise CustomException(e,sys)