import os
import sys
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import pandas as pd
from dotenv import load_dotenv
import pymysql
import pickle
import numpy as np

load_dotenv()

host= os.getenv('host')
user= os.getenv('user')
passw= os.getenv('password')
db= os.getenv('db')


def read_sql_data():
    logging.info("data reading initiated")
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=passw,
            database=db
        )
        logging.info('Connection Established',mydb)
        df = pd.read_sql_query('SELECT no_of_dependents, education,self_employed,income_annum,loan_amount,loan_term,cibil_score,residential_assets_value,commercial_assets_value,luxury_assets_value,bank_asset_value,loan_status FROM loan_approval_dataset',mydb)
        print(df.head())
        
        return df
        
    except Exception as e:
        raise CustomException(e,sys)
    


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)    