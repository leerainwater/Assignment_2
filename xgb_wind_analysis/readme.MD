# Homework Assignment 2 - BUAD689 Model Deployment
Lee Rainwater [lee.rainwater@tamu.edu](mailto:lee.rainwater@tamu.edu)

## Description
This repository contains working code for a Flask API which receives and responds to requests to predict against a model that is serialized into a Pickle file. A Dockerfile and structure are provided which can be used to generate a Docker image and a working container serving the code below.

## Contents
`WindFlaskTest.py` -- execute this to call the Flask API to demonstrate a prediction from the serialized model in the deployed Docker container. By default, this script queries the Docker container on port 8080.

`\src\LeeRainwaterXGBoost-RF_Pickle-flask.py` -- the Flask API that serves the model predictions

`\src\LeeRainwaterXGBoost-RF_Pickle.py` -- the script which generates the model and dumps it into a pickle file

`\src\static\altitude-pickleRF.dat` -- the pickle file containing the serialized model

`\src\static\TH210_final_wind_analysis-elr.csv` -- the raw data used to develop the model

To build the Docker container, cd to the directory containing the Dockerfile and type:
`docker build -t rainwater-hw-02 .`

To deploy the Docker container, type:
`docker run -d --rm -p 8080:8080 --name winds rainwater-hw-02`