#Takes the dataset of positives and removes irrelevant columns and irrelevant
#observation from healthy tissues


def select_columns_and_clean_iedb(data_frame):
    list_columns = ['Epitope - Name',
                'Epitope - Source Organism',
                'Epitope - Species',
                '1st in vivo Process - Process Type',
                '1st in vivo Process - Disease',
                '1st in vivo Process - Disease Stage',
                'Assay - Method',
                'Assay - Response measured',
                'Assay - Qualitative Measure',
                'Assay - Number of Subjects Tested',
                'Assay - Response Frequency (%)',
                'cell_type_in_assay',
                'MHC Restriction - Class']

    data_frame = data_frame[list_columns].drop_duplicates()
    data_frame = data_frame[
    (data_frame['1st in vivo Process - Process Type'] != "No immunization") |
    (data_frame['1st in vivo Process - Disease'] != "healthy")]

    return data_frame
