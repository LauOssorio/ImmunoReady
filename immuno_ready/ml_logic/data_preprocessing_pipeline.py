## STEPS
## 1. Data cleaning: clean amino acid sequences, remove too long or too short sequences
## 2. Prepare the data: add relevant columns in negative dataset, merge with positive dataset
## 3. Target engineering: calculate inmmunogenicity score and safety
## 4. Slice the datasets for pre-training tests - OPTIONAL
## 5. Construct AA matrix for processing, pad sequences

import pandas as pd



def prepare_training_set(data_positives, data_negatives, slice_pretraining = True):
    from  immuno_ready.ml_logic.add_relevant_columns import add_relevant_columns
    from immuno_ready.ml_logic.peptide_cleaning import peptide_cleaning
    from data_cleaning_tools import select_columns_and_clean_iedb, create_target_features

    data_positives_clean = select_columns_and_clean_iedb(data_positives)
    data_negatives_clean = add_relevant_columns(data_negatives)

    full_dataset = pd.concat([data_positives_clean, data_negatives_clean] ignore_index= True)

    dataset_target = 



def slice_pretraining(df):
