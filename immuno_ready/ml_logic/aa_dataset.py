# Data analysis
import pandas as pd
import numpy as np

# Data vis
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Misc
from pathlib import Path
from typing import Type, Dict, List

#Sklearn
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.decomposition import PCA


def build_dataset3(path_index_file):
    '''
    Constructs a unified pandas DataFrame from individual result text files in a folder,
    assigning measurement names from the associated AAindex-like reference file.

    Parameters
    ----------
    path_index_file : Path
        Path to the AAindex file, which contains measurement/reference names in lines starting with "D ".
        The file must be in the same folder as the individual result tables, which are .txt files in the same directory.

    Returns
    -------
    dataset3_df : pandas.DataFrame
        A DataFrame where each row corresponds to the data from a single result file,
        indexed by the file's stem (filename without extension) and columns corresponding to amino acids.
        Adds a "Parameters" column with the associated measurement/reference names (D-starting lines).
    '''

    # Extract paper and measurement names into a list
    index_file = open(path_index_file, "r").read()
    references = []
    for line in index_file.splitlines():
        if line.startswith("D "):
            ref = line[2:].strip()
            references.append(ref)


    # Extract data from the individual files, convert to numpy array
    txt_files = [str(file) for file in Path(path_index_file).parent.glob('*.txt')]
    txt_files.sort()

    arr_list = []
    for file in txt_files:
        row = pd.read_csv(file,sep='\s+', header=None).to_numpy().flatten()
        arr_list.append(row)

    numpy_aa_details = np.stack(arr_list)


    # Define Column and Row (Index) Names for the dataframe
    col_names = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    col_names.append('Parameters')

    index_names = [Path(txt).stem for txt in txt_files]

    # Build the DataFrame
    dataset3_df = pd.DataFrame(numpy_aa_details)
    dataset3_df['Parameters'] = references
    dataset3_df.columns = col_names
    dataset3_df.index = index_names

    return dataset3_df




def preprocess_dataset3(
    df: pd.DataFrame,
    simple_imputer_strategy: str = 'median',
    scaler: Type = MinMaxScaler
) -> pd.DataFrame:
    """
    Transpose and preprocess the dataset by imputing missing values and scaling features.

    Parameters
    ----------
    df : pd.DataFrame (output from the build_dataset3 function)

    simple_imputer_strategy : str, optional (default: 'median')
        Strategy used by `sklearn.impute.SimpleImputer` for filling missing values, e.g., 'mean', 'median', 'most_frequent'.

    scaler : Type, optional (default: MinMaxScaler)
        A scaler class (not an instance) from sklearn, e.g. MinMaxScaler or StandardScaler, used to scale the data.

    Returns
    -------
    pd.DataFrame
        The preprocessed (imputed and scaled) DataFrame with the same columns as the transposed input (excluding the last row).
    """

    df_transposed = df.T
    df_transposed = df_transposed.iloc[:-1]  # Drop last row, typically non-numeric such as 'Parameters'

    # Ensure all data is float for numerical processing
    df_transposed = df_transposed.astype('float')

    # Create a pipeline: impute missing values, then scale
    col_transformer = make_pipeline(
        SimpleImputer(strategy=simple_imputer_strategy),
        scaler()
    ).set_output(transform="pandas")

    # Apply the pipeline to the data
    df_transposed_processed = col_transformer.fit_transform(df_transposed)

    return df_transposed_processed


def pca_project_plotcumvar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Project the preprocessed dataframe onto its principal components using PCA,
    and transposed.
    Also plots the cumulative variance explained by the number of PCs.

    Parameters
    ----------
    df : pd.DataFrame, transposed output from the preprocess_dataset3 function

    Returns
    -------
    pd.DataFrame
        Data projected onto principal components.
    Also prints the explained cumulative PC variance.
    """

    pca = PCA().set_output(transform="pandas")
    dataset3_proj = pca.fit_transform(df)

    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Cumulative Explained Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return dataset3_proj.T


def plot_pca_proj(df: pd.DataFrame, three_dim: bool = False) -> None:
    """
    Visualize PCA results in 2D or 3D, colored by amino acid classification.

    Parameters
    ----------
    df : pd.DataFrame
        Output from pca_project function
    three_dim : bool, optional
        If True, creates a 3D scatter plot; otherwise, 2D (default: False).
    """
    AA_dict = {
        'A': 'Nonpolar (Hydrophobic)', 'V': 'Nonpolar (Hydrophobic)', 'L': 'Nonpolar (Hydrophobic)',
        'I': 'Nonpolar (Hydrophobic)', 'M': 'Nonpolar (Hydrophobic)', 'F': 'Aromatic', 'Y': 'Aromatic',
        'W': 'Aromatic', 'S': 'Polar, Uncharged', 'T': 'Polar, Uncharged', 'N': 'Polar, Uncharged',
        'Q': 'Polar, Uncharged', 'C': 'Polar, Uncharged', 'K': 'Positively Charged (Basic)',
        'R': 'Positively Charged (Basic)', 'H': 'Positively Charged (Basic)', 'D': 'Negatively Charged (Acidic)',
        'E': 'Negatively Charged (Acidic)', 'G': 'Special', 'P': 'Special'
    }

    df = df.T
    df["classification"] = df.index.map(AA_dict)

    if not three_dim:
        plt.figure(figsize=(7, 7))
        sns.scatterplot(
            x=df['pca0'],
            y=df['pca1'],
            hue=df["classification"],
            s=150,
            edgecolor="k",
            palette="Set2"
        )
        plt.show()
    else:
        fig = px.scatter_3d(
            df,
            x='pca0',
            y='pca1',
            z='pca2',
            color='classification',
            symbol='classification',
            size_max=15,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_traces(marker=dict(size=12, line=dict(width=1, color='black')))
        fig.update_layout(width=900, height=700, legend_title="Classification")
        fig.show()




def dataset3_process_all(path_index_file: str):
    """
    Builds, preprocesses, and performs PCA on dataset3, plotting the cumulative explained variance.

    Parameters
    ----------
    path_index_file : str
        Path to the index file for dataset3.

    Returns
    -------
    pd.DataFrame
        Data projected onto principal components.
    """
    df = build_dataset3(path_index_file)
    df_processed = preprocess_dataset3(df)
    pca = pca_project_plotcumvar(df_processed)
    return pca



def generate_matrix_for_peptide(
    peptide: str,
    pca_table: pd.DataFrame
) -> np.ndarray:
    """
    Converts a peptide sequence into a matrix using PCA vector encoding.

    Parameters
    ----------
    peptide : str
        Amino acid sequence.
    pca_table : pd.DataFrame
        DataFrame output from the PCA generation.

    Returns
    -------
    np.ndarray
        2D array where each row corresponds to the PCA vector of an amino acid.

    Raises
    ------
    KeyError
        If a character in peptide is not present in pca_table.
    """
    return np.stack([pca_table[aa] for aa in peptide])




def generate_matrices_for_dataset(
    dataset: pd.DataFrame,
    pca_table: pd.DataFrame
) -> List[np.ndarray]:
    """
    Applies PCA vector encoding to all peptide sequences in a dataset.

    Parameters
    ----------
    dataset : pd.DataFrame
        DataFrame containing a column 'Epitope - Name' with peptide sequences.
    pca_table : pd.DataFrame
        DataFrame output from the PCA generation.

    Returns
    -------
    List[np.ndarray]
        List of 2D arrays, each corresponding to a peptide.

    Raises
    ------
    KeyError
        If 'Epitope - Name' column is missing or contains unknown amino acids.
    """
    peptides = dataset['Epitope - Name'].tolist()
    return [generate_matrix_for_peptide(p, pca_table) for p in peptides]
