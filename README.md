Disclaimer: this repo was created for learning and teaching purposes in the context of a Data Science bootcamp. Results have not scientif value and code is not peer-reviewed. Do not use for research

# ImmunoReady
This repository provides a machine learning-based pipeline to predict the immunogenicity of peptides for use in **cancer vaccine development**.


## Preprocessing

We count with three datasets: one for peptides related to immune responses (from IEDB database, or positives), one for peptides from healthy tissues (from HLA-ligand human atlas, or negatives) and a third dataset with almost 600 physicochemical features of the 20 amino acids (AAindex one). The preprocesing has the following steps

1. Dataset positives:
In this first table, irrelevant features are dropped and a column "cell_type_in_assay" is added to state whether the peptides have been tested in the context of T or B lymphocytes immunity.

After downloading the dataset, we can run select_columns_and_clean_iedb from data_cleaning_tools.py

2. Dataset negatives:
In this second dataset, columns are armonised to become compatible with the first one.

After downloading the dataset, we can run add_relevant_columns from add_relevant_columns.py on the table and merge with the first one.

3. Safety and immunogenicity scores:
Feature engineering of the target classes. This step is done after merging the previous two tables. Details about how the columns are build can be foun in the docstring of the fucntion create_target_features in data_cleaning_tools.py


The actual code squema:
```

# load the first data set
df_1 = pd.read_csv('./raw_data/raw_positives_iedb.csv')

# Apply the first function
from  immuno_ready.ml_logic.data_cleaning_tools import select_columns_and_clean_iedb
df_1_filtered = select_columns_and_clean_iedb(df_1)

# Load the second data set
normals = pd.read_csv('./raw_data/raw_negatives_hla_ligand_atlas.tsv', sep='\t')

# Apply the second function
from  immuno_ready.ml_logic.add_relevant_columns import add_relevant_columns
normal_clean = add_relevant_columns(normals)

# Datasets ready to be merged - MERGEEEEE!!
full_df = pd.concat([df_1_filtered, normal_clean], ignore_index=True)

# Construct the targets on the full data set
from  immuno_ready.ml_logic.data_cleaning_tools import create_target_features
target_df = create_target_features(full_df)

# Clean the peptide sequences and remove those that are common in dataset 1 and dataset2 (we cannot interpret them)
from immuno_ready.ml_logic.peptide_cleaning import peptide_cleaning
clean_target_df = peptide_cleaning(target_df)

# Implement encoding of categorical features

´´´
