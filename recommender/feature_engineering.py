from .text_processing import process_texts

def add_tags(df):
    cols_to_process = ['Category', 'Brand', 'Description']

    # Process each column with NLP
    for col in cols_to_process:
        df[col] = process_texts(df[col].fillna("").astype(str).tolist())

    # Merge into Tags column
    df['Tags'] = df[cols_to_process].apply(lambda row: ", ".join(row), axis=1)
    return df
