# Function to calculate peptide length
def peptide_length(peptide):
        return len(str(peptide))

# Function that drops all peptides with sequence length > 25 (or another number) AA
def drop_large_sequences (data_frame , max_length):

    # Add peptide length column temporarily
    data_frame['peptide length'] = data_frame['Epitope - Name'].apply(peptide_length)

    # Drop rows with too long sequences
    data_frame = data_frame[data_frame['peptide length'] <= max_length]

    # Remove temporary column
    data_frame.drop(columns='peptide length', inplace=True)

    return data_frame


# Function that gets rid of negative peptides that also exist in the positive dataset
def remove_overlap(neg_df , pos_df):
    neg_df = neg_df[ ~ neg_df['Epitope - Name'].isin(pos_df['Epitope - Name'])]
    return neg_df
