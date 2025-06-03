def select_columns_and_clean_iedb(data_frame):
    """
    Filters and cleans an IEDB dataset by selecting relevant columns and removing
    observations that are not associated with immunization or disease.

    This function performs the following steps:
    1. Selects a predefined set of biologically relevant columns related to epitopes,
       assay conditions, and MHC restrictions.
    2. Drops duplicate rows to avoid redundant entries.
    3. Removes rows corresponding to observations from healthy tissues or cases with
       no immunization, as they are not informative for immunogenicity analysis.

    Parameters:
    ----------
    data_frame : pandas.DataFrame
        The input DataFrame containing IEDB data.

    Returns:
    -------
    pandas.DataFrame
        A cleaned DataFrame containing only relevant columns and filtered observations.
    """
    
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


# TODO: Check for weird characters in amino acid sequences
