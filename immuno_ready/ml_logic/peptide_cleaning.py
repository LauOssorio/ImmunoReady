
def fix_weird_peptides (data_frame):
# Remove weird peptides values

    # Remove the right-hand part for peptides that contain a " + "
    data_frame.loc[:,'Epitope - Name'] = data_frame['Epitope - Name'].str.split('+').str[0].str.strip()

    # Boolean mask to filter out sequences that contain "U"
    u_mask = data_frame['Epitope - Name'].str.contains('U')
    data_frame = data_frame[~u_mask]

    # Boolean mask to filter out sequences that contain "X"
    x_mask = u_mask = data_frame['Epitope - Name'].str.contains('X')
    data_frame = data_frame[~x_mask]

    return data_frame


def remove_overlap(combined_df):
# Remove NEGATIVE peptides that also appear in the positive dataset

    # Find peptides that exist in the positive dataset
    positive_peptides = combined_df.loc[combined_df['Assay - Qualitative Measure'] != 'Negative', 'Epitope - Name'].unique()

    # Remove rows in the above where 'Assay - Qual measure' is negative (ie negative peptides that also exist in the positive dataset)
    combined_df = combined_df[~((combined_df['Epitope - Name'].isin(positive_peptides)) & (combined_df['Assay - Qualitative Measure'] == 'Negative'))]

    return combined_df


def peptide_length(peptide):
# Function to calculate peptide length
        return len(str(peptide))


def drop_large_and_short_sequences (data_frame , min_length, max_length):
# Function that drops all peptides with sequence length > 25 (or any chosen max_length) AA
# FYI: choosing 25 removes c.7k rows (after removal of weird values)

    # Add peptide length column temporarily
    data_frame.loc[:,'peptide length'] = data_frame['Epitope - Name'].apply(peptide_length)

    # Drop rows with too long sequences
    data_frame = data_frame[data_frame['peptide length'] <= max_length]

    # Drop rows with too short sequences
    data_frame = data_frame[data_frame['peptide length'] >= min_length]

    # Remove temporary column
    data_frame.drop(columns='peptide length', inplace=True)

    return data_frame


def peptide_cleaning (data_frame , min_length =8, max_length = 25):
    # Final cleaning function

    # Remove / fix weird peptides
    data_frame = fix_weird_peptides (data_frame)

    # Remove overlap between neg and pos (only negatives are deleted)
    data_frame = remove_overlap (data_frame)

    # Remove unusually long sequences (default is >25 but any max_length will work)
    data_frame = drop_large_and_short_sequences (data_frame , min_length, max_length)

    return data_frame
