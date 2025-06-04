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
    data_frame = data_frame[data_frame["1st in vivo Process - Process Type"] != "No immunization"]
    data_frame = data_frame[data_frame["1st in vivo Process - Disease"] != "healthy"]

    return data_frame


# TODO: Check for weird characters in amino acid sequences

def create_target_features(data_frame, drop_intermediary_columns = True):
    """
    Generates target features related to peptide safety and immunogenic strength
    for use in predictive modeling or downstream analysis.

    This function:
    - Assigns a categorical safety label (`peptide_safety`) based on the
      '1st in vivo Process - Process Type' using predefined rules.
    - Aggregates safety information per peptide and retains the most conservative
      (highest risk) classification using a safety ranking system.
    - Assigns an immunogenicity strength score (`peptide_strength`) based on
      assay results and conditions.
    - Computes `target_strength`, a numeric score penalized by experimental
      robustness (number of subjects tested) and enhanced by immune cell type.

    Parameters
    ----------
    data_frame : pandas.DataFrame
        The input IEDB dataset with epitope and assay information.

    drop_intermediary_columns : bool, optional (default=True)
        Whether to drop intermediary calculation columns (e.g., safety_rank,
        peptide_strength, penalty_strength, etc.) from the final output.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame augmented with:
        - `peptide_safety` : categorical safety label.
        - `target_strength` : numeric immunogenicity score combining assay result,
                              number of tested subjects, and immune cell type.
        Intermediary columns are dropped unless `drop_intermediary_columns=False`.

    Notes
    -----
    - Safety labels are hard-coded based on expert-assigned mappings of process types.
    - If multiple entries exist for the same peptide, the most conservative
      (lowest safety_rank) one is retained.
    - Strength scores are heuristically assigned and should be treated as
      semi-quantitative.
    """
    import numpy as np
    ## Hard coding the decisions on safety scoring
    conditions_safety = [
        data_frame["1st in vivo Process - Process Type"] == 'Occurrence of infectious disease',
        data_frame["1st in vivo Process - Process Type"] =='Occurrence of allergy',
        data_frame["1st in vivo Process - Process Type"] =='Exposure with existing immune reactivity without evidence for disease',
        data_frame["1st in vivo Process - Process Type"] == 'Unknown',
        data_frame["1st in vivo Process - Process Type"] == 'Occurrence of autoimmune disease',
        data_frame["1st in vivo Process - Process Type"] =='Environmental exposure to endemic/ubiquitous agent without evidence for disease',
        data_frame["1st in vivo Process - Process Type"] =='Prophylactic vaccination',
        data_frame["1st in vivo Process - Process Type"] =='Administration in vivo',
        data_frame["1st in vivo Process - Process Type"] =='Occurrence of disease',
        data_frame["1st in vivo Process - Process Type"] =='Exposure without evidence for disease',
        data_frame["1st in vivo Process - Process Type"] =='Occurrence of cancer',
        data_frame["1st in vivo Process - Process Type"] =='Documented exposure without evidence for disease',
        data_frame["1st in vivo Process - Process Type"] =='Transplant/transfusion',
        data_frame["1st in vivo Process - Process Type"] =='Vaccination',
        data_frame["1st in vivo Process - Process Type"] =='Therapeutic vaccination',
        data_frame["1st in vivo Process - Process Type"] =='Administration in vivo to cause disease',
        data_frame["1st in vivo Process - Process Type"] == np.nan,
        data_frame["1st in vivo Process - Process Type"] =='Administration in vivo to prevent or reduce disease',
        data_frame["1st in vivo Process - Process Type"] == 'None'
        ]

    choices_safety = [
        "very safe",
        "safe",
        "very safe",
        "unknown",
        "very risky - autoimmunity",
        "very safe",
        "very safe",
        "safe",
        "safe",
        "safe",
        "safe",
        "safe",
        "risky",
        "very safe",
        "very safe",
        "very safe",
        "unknown",
        "very safe",
        "very safe"
    ]


    data_frame["peptide_safety"] = np.select(conditions_safety, choices_safety, default = "other")

    # Now, if a peptide has different values,
    # keep the most conservative one = the one that states more risk


    safety_rank = {
        "very risky - autoimmunity": 0,
        "risky": 1,
        "safe": 2,
        "very safe": 3,
        "unknown": 4,
        "other": 5
    }

    # Add numeric rank to help pick the most conservative
    data_frame["safety_rank"] = data_frame["peptide_safety"].map(safety_rank)

    # Now group by peptide_sequence and keep the row with the lowest (most conservative) rank
    data_frame = (
        data_frame.sort_values("safety_rank")
        .drop_duplicates(subset="Epitope - Name", keep="first")
    )

        # strength: from 0 to 10 (arbitrary)

    conditions_strength = [data_frame['1st in vivo Process - Disease'] == 'Healthy',
                        data_frame["Assay - Qualitative Measure"] == 'Positive',
                        data_frame["Assay - Qualitative Measure"] =='Positive-Low',
                        data_frame["Assay - Qualitative Measure"] =='Positive-Intermediate',
                        data_frame["Assay - Qualitative Measure"] =='Positive-High',
                        data_frame["Assay - Qualitative Measure"] =='Assay - Qualitative Measurement'
                        ]


    choices_strength = [
        0,
        7,
        3,
        8,
        10,
        5
    ]


    data_frame["peptide_strength"] = np.select(conditions_strength, choices_strength, default = np.nan)

    data_frame["averaged_strength"] = (
        data_frame.groupby("Epitope - Name")["peptide_strength"]
        .transform("mean")
    )
    # Creating a colum with the sum of the total individuals tested per peptide

    data_frame["averaged_number_subjects_tested"] = (
        data_frame.groupby("Epitope - Name")["Assay - Number of Subjects Tested"]
        .transform(lambda x: x.sum(min_count=1)))
    # Number of experiments - penalising strength score

    conditions_penal = [data_frame['1st in vivo Process - Disease'] == 'Healthy',
                        data_frame["averaged_number_subjects_tested"] <3 ,
                        (data_frame["averaged_number_subjects_tested"] >= 4)
                                & (data_frame["averaged_number_subjects_tested"] < 10),
                        (data_frame["averaged_number_subjects_tested"] >=10)
                                & (data_frame["averaged_number_subjects_tested"] < 30),
                            (data_frame["averaged_number_subjects_tested"] >=30),
                            data_frame["averaged_number_subjects_tested"].isna()
                        ]

    choices_penal = [
        0,
        0.5,
        0.6,
        0.85,
        1,
        0.70
    ]

    data_frame["penalty_strength"] = np.select(conditions_penal, choices_penal, default = np.nan)
    conditions_cells = [data_frame["cell_type_in_assay"] == "None" ,
                        data_frame["cell_type_in_assay"] == "T cells lymphocytes" ,
                       data_frame["cell_type_in_assay"] == "B cells lymphocytes"
                       ]

    choices_cells = [
        0,
        1,
        0.5,
    ]

    data_frame["plus_cells"] = np.select(conditions_cells, choices_cells, default = np.nan)

    data_frame["target_strength"] = (data_frame["peptide_strength"] *
                                     data_frame["penalty_strength"] *
                                     data_frame["plus_cells"])

    # filter the unkown out
    data_frame = data_frame[data_frame["peptide_safety"] != "unknown"]
    data_frame = data_frame[data_frame["peptide_safety"] != "other"]

    if drop_intermediary_columns == True:
        data_frame = data_frame.drop(['safety_rank',
                                      'peptide_strength',
                                      'averaged_strength',
                                      'averaged_number_subjects_tested',
                                      'penalty_strength',
                                      'plus_cells'],
                                     axis=1)

        return data_frame
    else:
        return data_frame
