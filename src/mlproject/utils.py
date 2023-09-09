import os
import sys
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import pandas as pd
from dotenv import load_dotenv
import pymysql

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
        df = pd.read_sql_query('select * from loan_approval_dataset',mydb)
        print(df.head())
        
        return df
        
    except Exception as e:
        raise CustomException(e,sys)