## STEPS
## 1. Data cleaning: clean amino acid sequences, remove too long or too short sequences
## 2. Prepare the data: add relevant columns in negative dataset, merge with positive dataset
## 3. Target engineering: calculate inmmunogenicity score and safety
## 4. Slice the datasets for pre-training tests - OPTIONAL
## 5. Construct AA matrix for processing, pad sequences

import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from add_relevant_columns import *
from peptide_cleaning import *
from data_cleaning_tools import *
from aa_dataset import *
import os



def slice_pretraining_func(df, number_of_observations):
    """
    This function slices a given number of observations from the full preprocessed training dataset,
    balancing safe peptides across disease types and including risky peptides from a single species.
    If the number of onbservations provided is higher than the smallest category in the training set
    (risky peptides), then the smallest of both is chosen for doing the sample slicing.

    Parameters:
        df (pd.DataFrame): Full dataset with 'peptide_safety' and disease columns.
        number_of_observations (int): Number of total rows to return (balanced between safe and risky).

    Returns:
        pd.DataFrame: Sampled and balanced subset of the original dataframe.
    """

    #starting from the full data set
    safe_df = df[df['peptide_safety'] == 0]
    risky_df = df[df['peptide_safety'] == 1]

    max_safe = min(len(risky_df), int(number_of_observations))

    n_diseases = (safe_df["disease_group"].nunique())
    max_safe_per_disease_group = int(max_safe // n_diseases)

  # Group safe peptides by type of disease
    safe_disease_groups = safe_df.groupby("disease_group")

        # Resample from each disease group
    balanced_safe = pd.concat([
        resample(group, replace=False, n_samples = max_safe_per_disease_group, random_state = 42)
        for _, group in safe_disease_groups #ignoring the name of the group from GroupBy object
    ])

        # ---- Sample risky peptides ----
    sampled_risky = risky_df.sample(n = max_safe, random_state=42)

    # ---- Combine ----
    final_df = pd.concat([balanced_safe, sampled_risky], ignore_index=True)

    return final_df



def balance_prepare_training_set(data_positives, data_negatives,balance_classes=True, number_of_observations = 6000):

    """
    Combines, balances, cleans, and processes positive and negative peptide datasets
    into a unified training set with target features.
    If the whole dataset is to be used, set the number of observations to 1000000000.

    This function:
    - Cleans and standardizes the input positive and negative datasets.
    - Merges them into a single DataFrame.
    - Applies peptide sequence cleaning and filtering.
    - Generates target features for downstream modeling.
    - Slices the dataset for pretraining.

    Parameters
    ----------
    data_positives : pd.DataFrame
        DataFrame containing positive samples (e.g. immunogenic peptides).
    data_negatives : pd.DataFrame
        DataFrame containing negative samples (e.g. non-immunogenic peptides).
    number_of_observations : int, optional
        Number of samples to return if slicing for pretraining. Default is `len_`, which
        should be defined in the broader scope.

    Returns
    -------
    pd.DataFrame
        A cleaned and processed dataset with target features ready for training or pretraining.

    Notes
    -----
    - Uses several utility functions from `immuno_ready.ml_logic` and `data_cleaning_tools`.
    - Assumes global variables `slice_pretraining` and `slice_pretraining_func` are defined
      externally to control optional slicing.
    - Input DataFrames should conform to expected schema for the cleaning and processing functions.
    """



    data_positives = select_columns_and_clean_iedb(data_positives)
    data_negatives = add_relevant_columns(data_negatives)

    full_dataset = pd.concat([data_positives, data_negatives], ignore_index= True)

    # Cleaning peptides sequences and filtering long and short ones

    full_dataset_clean = peptide_cleaning(full_dataset)

    # Create target features

    dataset_target = create_target_features(full_dataset_clean)

    if balance_classes ==True:

        balanced_training_df = slice_pretraining_func(dataset_target, number_of_observations)

        return balanced_training_df
    else:
        return dataset_target





parent_path = os.getcwd()

path_1 = os.getcwd()
parent_path = os.path.dirname(path_1)
final_path_pca = parent_path + "/data/dataset3_pca.csv"


def split_training_set(dataset_target,
                           validation_size = 0.3,
                           random_state = 42,
                           pca_table_path = final_path_pca,
                           #normalization_method = "MinMaxScaler()"
                           ):

    """
    Preprocesses a peptide dataset by splitting it into training and validation sets,
    and generating feature matrices using PCA-based encoding.

    This function:
    - Splits the dataset into training and validation sets for both classification
      (`peptide_safety`) and regression (`target_strength`) targets.
    - Generates numerical feature matrices for model input using a PCA transformation table.
    - Pads the peptide sequences to the maximun length

    Parameters
    ----------
    dataset_target : pd.DataFrame
        Input DataFrame containing peptide features and target labels.
        Must include columns: 'peptide_safety', 'target_strength', and 'disease_group'.
    validation_size : float, optional (default=0.3)
        Proportion of the dataset to include in the validation split.
    random_state : int, optional (default=42)
        Random seed used to ensure reproducibility of the train/validation split.
    pca_table_path : str, optional (default="./immuno_ready/data/dataset3_pca.csv")
        File path to the PCA transformation table used to encode peptide sequences.

    Returns
    -------
    y_train_class : pd.Series
        Training labels for classification (peptide safety).
    y_val_class : pd.Series
        Validation labels for classification (peptide safety).
    y_train_reg : pd.Series
        Training labels for regression (target strength).
    y_val_reg : pd.Series
        Validation labels for regression (target strength).
    X_train_matrix : np.ndarray or pd.DataFrame
        Encoded feature matrix for training data.
    X_val_matrix : np.ndarray or pd.DataFrame
        Encoded feature matrix for validation data.

    Notes
    -----
    This function requires the `generate_matrices_for_dataset` function from the
    `aa_dataset` module to perform peptide encoding based on the PCA table.

    Examples
    -----
    >>> y_train_class, y_val_class, y_train_reg, y_val_reg, X_train_matrix_pad, X_val_matrix_pad = split_training_set(balances_training_set)

    """

    pca_table = pd.read_csv(pca_table_path)

    X = dataset_target.drop(columns=['peptide_safety', 'target_strength', 'disease_group'])
    y_class = dataset_target['peptide_safety']
    y_reg = dataset_target['target_strength']


    X_train, X_val, y_train_class, y_val_class = train_test_split(X, y_class, test_size = 0.3, random_state = 42)
    X_train, X_val, y_train_reg, y_val_reg = train_test_split(X, y_reg, test_size = 0.3, random_state = 42)



    X_train_matrix = generate_matrices_for_dataset(X_train, pca_table)
    X_val_matrix = generate_matrices_for_dataset(X_val, pca_table)

    X_train_matrix_pad = pad_sequences(X_train_matrix, dtype='float32', padding='post')
    X_val_matrix_pad = pad_sequences(X_val_matrix, dtype='float32', padding='post')

    return y_train_class, y_val_class, y_train_reg, y_val_reg, X_train_matrix_pad, X_val_matrix_pad
