import os
import sys
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import pandas as pd
from src.mlproject.utils import read_sql_data
from sklearn.model_selection import train_test_split


from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_cofig = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        try:
            df=read_sql_data()
            logging.info('Reading data from database has complete')
            
            os.makedirs(os.path.dirname(self.ingestion_cofig.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_cofig.raw_data_path,index=False,header=True)
            train_set,test_set= train_test_split(df,test_size=0.25,random_state=42)
            train_set.to_csv(self.ingestion_cofig.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_cofig.test_data_path,index=False,header=True)
            
            logging.info("Data Ingestion Completed")
            
            return(
                self.ingestion_cofig.train_data_path,
                self.ingestion_cofig.test_data_path
            )
            
        
        
        
        except Exception as e:
            raise CustomException(e,sys)
                
        

