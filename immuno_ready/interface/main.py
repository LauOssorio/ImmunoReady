"""
MAIN script of the project.
3 functions:
-preprocess
-train
-pred
"""

import pandas as pd
import numpy as np
from typing import AnyStr
from sklearn.model_selection import train_test_split
from keras import preprocessing

import os
import sys

sys.path.append(os.path.abspath('..'))
from immuno_ready.params import *
from immuno_ready.ml_logic.data_cleaning_tools import select_columns_and_clean_iedb, create_target_features
from immuno_ready.ml_logic.add_relevant_columns import add_relevant_columns
from immuno_ready.ml_logic.peptide_cleaning import peptide_cleaning
from immuno_ready.ml_logic.ds1_finalsteps import final_preproc, ohe
from immuno_ready.ml_logic.aa_dataset import generate_matrix_for_peptide, generate_matrices_for_dataset
from immuno_ready.ml_logic.models import initialize_LSTM, compile_LSTM, fit_LSTM
from immuno_ready.ml_logic.registry import save_results, save_model, load_model
from immuno_ready.ml_logic.data_preprocessing_pipeline import slice_pretraining_func, balance_prepare_training_set, split_training_set

#"../raw_data/Dataset3_Github"

def preprocess(
    pos_csv_path:str = "raw_data/raw_positives_iedb.csv",
    neg_csv_path:str = "raw_data/raw_negatives_hla_ligand_atlas.tsv",
    ds2_path:str = "immuno_ready/data/dataset3_pca.csv",
    split_ratio: float = 0.02
    ):
    """
    """
    # Load files / paths
    data_positives = pd.read_csv(pos_csv_path)
    data_negatives = pd.read_csv(neg_csv_path,sep = '\t')

    # Split final test set
    data_positives_train, data_positives_test = train_test_split(data_positives, test_size=0.1, random_state=42)
    data_negatives_train, data_negatives_test = train_test_split(data_negatives, test_size=0.1, random_state=42)

    # Prepare balanced train sets
    training_set = balance_prepare_training_set(
        data_positives_train,
        data_negatives_train,
        balance_classes=True,
        number_of_observations = 29_440)

    y_train_class, y_val_class, y_train_reg, y_val_reg, X_train_matrix_pad, X_val_matrix_pad = split_training_set(
        training_set,
        pca_table_path = ds2_path)

    # Prepare final test set
    testing_set = balance_prepare_training_set(
        data_positives_test,
        data_negatives_test,
        balance_classes=False,
        number_of_observations = 1_000_000_000)
    X_test = testing_set.drop(columns=['peptide_safety', 'target_strength', 'disease_group'])
    X_test_matrix = generate_matrices_for_dataset(X_test, pd.read_csv(ds2_path))
    X_test_matrix_pad = preprocessing.sequence.pad_sequences(X_test_matrix, dtype='float32', padding='post')


    y_test_class = testing_set['peptide_safety']
    y_test_reg = testing_set['target_strength']

    # selecting nb of PC's
    X_train_matrix_pad_cut = X_train_matrix_pad[:, :, :14]
    X_val_matrix_pad_cut = X_val_matrix_pad[:, :, :14]
    X_test_matrix_pad_cut = X_test_matrix_pad[:, :, :14]

    # Return values
    X = [X_train_matrix_pad_cut, X_val_matrix_pad_cut, X_test_matrix_pad_cut]
    y_class = [y_train_class, y_val_class, y_test_class]
    y_reg = [y_train_reg, y_val_reg, y_test_reg]

    print("\n✅ preprocess done: ", X[0].shape, y_class[0].shape, "\n")
    return X, y_class, y_reg



def train_LSTM(
        learning_rate=0.001,
        batch_size=32,
        patience=10
    ) -> float:
    """
    """
    X, y_class, y_reg = preprocess()

    X_train_matrix_pad, X_val_matrix_pad = X[0], X[1]
    y_train, y_val = y_class[0], y_class[1]

    model = initialize_LSTM((X_train_matrix_pad.shape[1], X_train_matrix_pad.shape[2]))
    model = compile_LSTM(model=model)
    model, history = fit_LSTM(
        model=model,
        X=X_train_matrix_pad,
        y=y_train,
        validation_data=(X_val_matrix_pad, y_val)
    )

    val_accuracy = np.max(history.history['val_accuracy'])

    params = dict(
        context="train",
        row_count=len(X_train_matrix_pad)
    )

    # Save results on the hard drive
    save_results(params=params, metrics=dict(accuracy=val_accuracy))
    # Save model weight on the hard drive and on GCS
    save_model(model=model)

    print("\n✅ fit done:\n")
    return val_accuracy

def pred(
    peptid: AnyStr = None,
    ds2_path:str = "immuno_ready/data/dataset3_pca.csv"
    ) -> np.ndarray:
    """
    Make a prediction for a peptid using the latest trained model
    """
    model = load_model()
    assert model is not None

    X_pred = generate_matrix_for_peptide(peptid, pd.read_csv(ds2_path))
    X_pred_pad = preprocessing.sequence.pad_sequences(
        sequences=X_pred,
        maxlen=25,
        dtype='float32',
        padding='post')
    X_pred_pad = np.expand_dims(X_pred_pad, axis=0)
    X_pred_pad_cut = X_pred_pad[:, :, :14]
    y_pred = model.predict(X_pred_pad_cut)

    print("\n✅ prediction done: ", type(float(y_pred[0][0])), "\n")
    return y_pred[0][0]
