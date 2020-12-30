# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 17:00:03 2020

@author: pkmis
"""


import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import joblib
import jsonify
import sklearn
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = joblib.load("Mark_predicted_model.pkl")
@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    # global df
    
    input_features = [float(x) for x in request.form.values()]
    features_value = np.array(input_features)
    
    #validate input hours
    if input_features[0] <0 or input_features[0] >24:
        return render_template('index.html', prediction_text='Please enter valid hours between 1 to 24 if you live on the Earth')
        

    output = model.predict([features_value])[0][0].round(2)

    # input and predicted value store in df then save in csv file
    # df= pd.concat([df,pd.DataFrame({'Study Hours':input_features,'Predicted Output':[output]})],ignore_index=True)
    # print(df)   
    # df.to_csv('student_info.csv')

    return render_template('index.html', prediction_text='You will get [{}%] marks, when you do study [{}] hours per day '.format(output, int(features_value[0])))


if __name__ == "__main__":
    app.run(debug=True)