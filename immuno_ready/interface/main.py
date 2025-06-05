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

import os
import sys

sys.path.append(os.path.abspath('..'))
from immuno_ready.ml_logic.data_cleaning_tools import select_columns_and_clean_iedb, create_target_features
from immuno_ready.ml_logic.add_relevant_columns import add_relevant_columns
from immuno_ready.ml_logic.peptide_cleaning import peptide_cleaning
from immuno_ready.ml_logic.ds1_finalsteps import final_preproc, ohe
from immuno_ready.ml_logic.aa_dataset import *


# Load the first data set & apply cleaning fonction
pos_df = pd.read_csv("../raw_data/raw_positives_iedb.csv")
pos_df = select_columns_and_clean_iedb(pos_df)

# Load the second data set & apply cleaning fonction
neg_df = pd.read_csv("../raw_data/raw_negatives_hla_ligand_atlas.tsv",sep = '\t')
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
path = '../raw_data/Dataset3_Github'
pca = dataset3_process_all(path)

# Not padded peptide matrices
input2 = generate_matrices_for_dataset(final_df, pca)
