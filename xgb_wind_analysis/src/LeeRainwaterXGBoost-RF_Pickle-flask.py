#
# Author: Lee Rainwater
# Heavily Borrowed From: Jamey Johnston
# Title: SciKit Learn Example with pickle
# Date: 2020/01/30
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
from settings import APP_ROOT
import numpy as np
from numpy import loadtxt, vstack, column_stack
import xgboost
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Import pickle to save ML models
import pickle

# APP_ROOT = "/home/edwardrainwater/Documents/tamu/modelDeploy/hw01"
APP_ROOT = ""
filename = "TH210_final_wind_analysis-elr.csv"
filespec = os.path.join(APP_ROOT, filename)
# %%
# Load the Altitude Data
dataset = np.loadtxt(filespec, delimiter=',', skiprows=1, usecols=(0, 1, 2, 5))
# %%
# Headers of Data
headers = np.loadtxt(filespec, delimiter=',', skiprows=0, usecols=(0, 1, 2, 5), max_rows=1, dtype=str)
# "fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"
# %% Split the altitude data into X (independent variable) and y (dependent variable)
X = dataset[:,0:dataset.shape[1]-1].astype(float)
Y = dataset[:,dataset.shape[1]-1].astype(float)
# %% Split altitude data into train and validation sets
seed = 7
test_size = 0.3
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

# %% Fit model on altitude Training Data using eXtended Gradient Boosting
modelXGB = xgboost.XGBRegressor()
modelXGB.fit(X_train, y_train)

# %% Make predictions for Validation data
y_predXGB = modelXGB.predict(X_valid)
predictionsXGB = [round(value) for value in y_predXGB]

# %% Evaluate predictions
score = modelXGB.score(X_train, y_train)
print(f"Training score: {score:.3f}")
# %%
mseXGB = mean_squared_error(y_valid, y_predXGB)
print("Mean Squared Error of Prediction: %.2f%%" % (mseXGB))
# %% Create Dataset with Prediction and Inputs
predictionResultXGB = column_stack(([X_valid, vstack(y_valid), vstack(y_predXGB)]))

# %% Fit model on altitude Training Data using Random Forest save model to Pickle file
modelRF = RandomForestRegressor()
modelRF.fit(X_train, y_train)

# %% Make predictions for Validation data
y_predRF = modelRF.predict(X_valid)

# Evaluate predictions
mse_RF = mean_squared_error(y_valid, y_predRF)
print("Accuracy of Random Forest: %.2f%%" % (mse_RF))

# Create Dataset with Prediction and Inputs
predictionResultRF = column_stack(([X_valid, vstack(y_valid), vstack(y_predRF)]))

# save model to file
pickle.dump(modelRF, open("altitude-pickleRF.dat", "wb"))

# Load model from Pickle file
loaded_modelRF = pickle.load(open("altitude-pickleRF.dat", "rb"))

# Predict a altitude diversity from inputs
alt, lat, mo = (55, 20, 6)
alt_div = loaded_modelRF.predict([[alt, lat, mo]])

print(f'Using altitude {alt}kft, latitude {lat}, month {mo}')
print(f'Predicted altitude diversity: {alt_div[0]:.3f}')

# %%
