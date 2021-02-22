#
# Author: Lee Rainwater
# Heavily Borrowed From: Jamey Johnston
# Title: HW02 - Model prediction with Flask
# Date: 2020/02/21
# Email: lee.rainwater@tamu.edu
# Texas A&M University - MS in Analytics - Mays Business School
#

# Train models for stratospheric wind diversity by altitude, month, latitude
# Save model to file using pickle
# Load model and make predictions
#
# %%
# Import OS and set CWD
import os
from flask import Flask, request, redirect, url_for, flash, jsonify
from settings import APP_STATIC
import numpy as np
import pandas as pd
from numpy import loadtxt, vstack, column_stack
import xgboost
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle, json

app = Flask(__name__)

@app.route('/')
def predict():
    model_dump = "altitude-pickleRF.dat"
    # Load model from Pickle file
    model_dump = os.path.join(APP_STATIC, "altitude-pickleRF.dat")
    loaded_modelRF = pickle.load(open(model_dump, "rb"))

    jsonfile = request.get_json()
    data = pd.read_json(json.dumps(jsonfile), orient='index')
    print(data)

    result = dict()

    X = np.array(data[['alt', 'lat', 'mo']])

    print(X)

    prediction = loaded_modelRF.predict(X)

    for i in range(len(prediction)):
        result[i] = prediction[i]

    return(jsonify(result))

    if __name__ == '__main__':
        app.run(host='0.0.0.0', debug=True, port=8080)

    # Predict a altitude diversity from inputs
    alt, lat, mo = (55, 20, 6)
    alt_div = loaded_modelRF.predict([[alt, lat, mo]])

    print(f'Using altitude {alt}kft, latitude {lat}, month {mo}')
    print(f'Predicted altitude diversity: {alt_div[0]:.3f}')


