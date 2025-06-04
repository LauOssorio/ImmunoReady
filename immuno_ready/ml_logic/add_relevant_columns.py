import numpy as np
import pandas as pd

def add_relevant_columns (data_frame):
#Function that creates the same columns in the negative dataset as in the clean positive one
#(ie the positive dataset after applying the select_columns_and_clean_iedb function

    new_data_frame = pd.DataFrame()

    #Rename columns (peptide-name and MHC restriction class)
    data_frame.rename(columns = {'peptide_sequence':'Epitope - Name','hla_class':'MHC Restriction - Class'})

    #Create healthy/control columns
    new_data_frame['Epitope - Name'] = data_frame['peptide_sequence']
    new_data_frame['Epitope - Source Organism'] = 'Homo sapiens'
    new_data_frame['Epitope - Species'] = 'Homo sapiens'
    new_data_frame['1st in vivo Process - Process Type'] = 'None'
    new_data_frame['1st in vivo Process - Disease'] = 'Healthy'
    new_data_frame['1st in vivo Process - Disease Stage'] = 'Healthy'
    new_data_frame['Assay - Method'] = 'None'
    new_data_frame['Assay - Response measured'] = 'None'
    new_data_frame['Assay - Qualitative Measure'] = 'Negative'
    new_data_frame['Assay - Number of Subjects Tested'] = np.nan
    new_data_frame['Assay - Response Frequency (%)'] = np.nan
    new_data_frame['cell_type_in_assay'] = 'None'

    #Rename values in the hla_class column
    MHC_restriction_map = {
    'HLA-I': 'I',
    'HLA-II': 'II',
    'HLA-I+II': 'non classical'
    }

    new_data_frame['MHC Restriction - Class'] = data_frame['hla_class'].map(MHC_restriction_map)
    return new_data_frame
