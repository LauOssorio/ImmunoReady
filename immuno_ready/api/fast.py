"""
script for designing the API predict endpoint
"""

# BEFORE LAUNCHING CHECK FOR REQUIREMENTS

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allowing all middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"])  # Allows all headers


@app.get("/")
def root():
    return {
    'ping': 'hello'
}

# app.state.model = load_model()
# @app.get("/predict")
# def predict(dataframe):

#     model = app.state.model

#     return dict()
