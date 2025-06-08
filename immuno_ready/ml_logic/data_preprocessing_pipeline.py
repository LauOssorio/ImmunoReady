## STEPS
## 1. Data cleaning: clean amino acid sequences, remove too long or too short sequences
## 2. Prepare the data: add relevant columns in negative dataset, merge with positive dataset
## 3. Target engineering: calculate inmmunogenicity score and safety
## 4. Slice the datasets for pre-training tests - OPTIONAL
## 5. Construct AA matrix for processing, pad sequences

import pandas as pd
import numpy as np
from sklearn.utils import resample


def slice_pretraining_func(df, number_of_observations = 10000):
 # This function slices a given number of observations from the full preprocessed training dataset,
 # and it balances the peptides based on their origin and category.
    n_observations = number_of_observations
    n_safe = n_observations // 2
    n_risky = n_observations - n_safe


    safe_df = df[df['peptide_safety'] == 0]
    risky_df = df[df['peptide_safety'] == 1]

  # Group safe peptides by type of disease
    safe_disease_groups = safe_df.groupby("1st in vivo Process - Process Type")
    n_diseases = safe_disease_groups.ngroups
    samples_per_disease = n_safe//n_diseases

    # Resample from each disease group
    balanced_safe = pd.concat([
        resample(group, replace=False, n_samples=samples_per_disease, random_state=42)
        for _, group in safe_disease_groups #ignoring the name of the group from GroupBy object
    ])


        # ---- Sample risky peptides ----
    sampled_risky = risky_df.sample(n=n_risky, random_state=42)

    # ---- Combine ----
    final_df = pd.concat([balanced_safe, sampled_risky], ignore_index=True)
    return final_df



def prepare_training_set(data_positives, data_negatives, slice_pretraining = True):
    from  immuno_ready.ml_logic.add_relevant_columns import add_relevant_columns
    from immuno_ready.ml_logic.peptide_cleaning import peptide_cleaning
    from data_cleaning_tools import select_columns_and_clean_iedb, create_target_features

    data_positives = select_columns_and_clean_iedb(data_positives)
    data_negatives = add_relevant_columns(data_negatives)

    full_dataset = pd.concat([data_positives, data_negatives], ignore_index= True)

    # Cleaning peptides sequences and filtering long and short ones

    full_dataset_clean = peptide_cleaning(full_dataset)

    # Create target features

    dataset_target = create_target_features(full_dataset_clean)

    if slice_pretraining:
        return slice_pretraining_func(dataset_target)

    else:
        return dataset_target
