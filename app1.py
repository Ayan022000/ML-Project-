from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.mlproject.pipelines.prediction_pipeline import CustomData,PredictPipeline

application= Flask(__name__)

app= application
@app.route('/')
def index():
     return render_template('index.html')
 
@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
     if request.method== "GET":
         return render_template('home.html')
     else:
         data = CustomData(
            no_of_dependents = request.form.get(' no_of_dependents'),
            education = request.form.get(' education'),
            self_employed = request.form.get(' self_employed'),
            income_annum = request.form.get(' income_annum'),
            loan_amount = request.form.get(' loan_amount'),
            loan_term = request.form.get(' loan_term'),
            cibil_score = request.form.get(' cibil_score'),
            residential_assets_value = request.form.get(' residential_assets_value'),
            commercial_assets_value = request.form.get(' commercial_assets_value'),
            luxury_assets_value = request.form.get(' luxury_assets_value'),
            bank_asset_value = request.form.get(' bank_asset_value')

             )
         pred_df = data.get_data_as_data_frame()
         print(pred_df)
         
         predict_pipeline = PredictPipeline()
         results= predict_pipeline.predict(pred_df)
         return render_template('home.html',results=results[0])
     
     
if __name__ == "__main__" :
      app.run(host="0.0.0.0",debug=True)  
     