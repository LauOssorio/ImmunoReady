"""
script for designing the API predict endpoint
"""

import numpy as np
import pandas as pd
from keras import preprocessing
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from immuno_ready.ml_logic.registry import load_model
from immuno_ready.ml_logic.aa_dataset import generate_matrix_for_peptide

app = FastAPI()
app.state.model = load_model()

# Allowing all middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"])  # Allows all headers


@app.get("/")
def root():
    return {'Welcome to': 'peptide classification'}

class PredictionRequest(BaseModel):
    peptides: List[str]

@app.post("/predict")
def predict(request: PredictionRequest) -> dict:

    model = app.state.model
    assert model is not None

    results = []

    ds2_path = "immuno_ready/data/dataset3_pca.csv"
    df = pd.read_csv(ds2_path)

    for peptide in request.peptides:
        X_pred = generate_matrix_for_peptide(peptide, df)
        X_pred_pad = preprocessing.sequence.pad_sequences(
            sequences=X_pred,
            maxlen=25,
            dtype='float32',
            padding='post')
        X_pred_pad = np.expand_dims(X_pred_pad, axis=0)
        X_pred_pad_cut = X_pred_pad[:, :, :14]
        y_pred = model.predict(X_pred_pad_cut)
        prob = float(y_pred[0][0])

        if prob < 0.5:
            prediction = "Peptide is safe"
        else:
            prediction = "Peptide is dangerous, may trigger auto immune response"

        results.append({
            "peptide": peptide,
            "predicted_class": prediction,
            "probality_of_unsafe": f"{round(prob*100, 2)} %"
        })

    return dict(predictions=results)
