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


TODO:
- Clean peptides with some more characters apart from the 20 amino acids.
- Clean peptides with long sequences (>23 amino acids)
- Remove peptides that have non-canonical amino acids such as "U"
