"""
MAIN script of the project.
4 functions:
-preprocess
-train
-evaluate
-pred
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential, Input, layers, optimizers

import os
import sys

sys.path.append(os.path.abspath('..'))
from immuno_ready.ml_logic.data_cleaning_tools import select_columns_and_clean_iedb, create_target_features
from immuno_ready.ml_logic.add_relevant_columns import add_relevant_columns
from immuno_ready.ml_logic.peptide_cleaning import peptide_cleaning
from immuno_ready.ml_logic.ds1_finalsteps import final_preproc, ohe
from immuno_ready.ml_logic.aa_dataset import *
from immuno_ready.ml_logic.models import *
from immuno_ready.ml_logic.registry import save_model, load_model


def preprocess(
    pos_csv_path:str = "../raw_data/raw_positives_iedb.csv",
    neg_csv_path:str = "../raw_data/raw_negatives_hla_ligand_atlas.tsv",
    ds2_path:str = "../raw_data/Dataset3_Github"
    ):
    # Load the first data set & apply cleaning fonction
    pos_df = pd.read_csv(pos_csv_path)
    pos_df = select_columns_and_clean_iedb(pos_df)

    # Load the second data set & apply cleaning fonction
    neg_df = pd.read_csv(neg_csv_path,sep = '\t')
    neg_df = add_relevant_columns(neg_df)

    # Merging datasets & create scoring
    merge_df = pd.concat([pos_df, neg_df], ignore_index=True)
    full_df = create_target_features(merge_df)

    # Clean the peptide sequences and remove those that are common in dataset 1 and dataset2 (we cannot interpret them)
    clean_df = peptide_cleaning(full_df)

    # final data preprocessing
    final_df = final_preproc(clean_df)

    # OHE for model ready dataset 1 (input1 and y in returned dataframe)
    model_df = ohe(final_df)


    # dataset 2 preprocessing
    pca = dataset3_process_all(ds2_path)

    # Not padded peptide matrices
    input2 = generate_matrices_for_dataset(final_df, pca)

    risky_veryrisky = model_df[model_df['peptide_safety'].isin([0, 2])].sample(n=5000, random_state=42)
    safe_verysafe = model_df[model_df['peptide_safety'].isin([1, 3])].sample(n=5000, random_state=42)
    subset_balances = pd.concat([safe_verysafe, risky_veryrisky], ignore_index=True, axis=0)

    subset_balances['peptide_safety_binary'] = subset_balances['peptide_safety'].isin([0, 2]).astype(int)

    X = subset_balances.drop(columns=['peptide_safety', 'target_strength'])
    y = subset_balances[['peptide_safety', 'target_strength', 'peptide_safety_binary']]





def train_LSTM_model(df):



    # Create (X_train_processed, y_train, X_val_processed, y_val)

    X_train, X_val, y_train, y_val = train_test_split(X, y['peptide_safety_binary'], test_size=0.3, random_state=42)

    X_train_matrix = generate_matrices_for_dataset(X_train, pca.iloc[:14,:])
    X_val_matrix = generate_matrices_for_dataset(X_val, pca.iloc[:14,:])

    X_train_matrix_pad = pad_sequences(X_train_matrix, dtype='float32', padding='post')
    X_val_matrix_pad = pad_sequences(X_val_matrix, dtype='float32', padding='post')


    model = simple_LSTM()

    adam_opt = optimizers.Adam(learning_rate=0.001)

    model.compile(loss='binary_crossentropy',
              optimizer=adam_opt,
              metrics=['accuracy'])

    es = EarlyStopping(patience=10, restore_best_weights=True)

    history = model.fit(X_train_matrix_pad, y_train,
            epochs=100,  # Use early stopping in practice
            batch_size=32,
            verbose=1,
            validation_data = (X_val_matrix_pad, y_val),
            callbacks = [es])

    return model

def save_LSTM_model(train_data):
    model = train_LSTM_model(train_data)
    save_model(model)
    return None
