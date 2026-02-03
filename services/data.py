import pandas as pd

def load_data(file_path):
    #Loading the CSV file into into a pandas DataFrame
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from {file_path}")
    return df

def get_X_y(df, target_column):
    #Preparing features (X) and target (y) for model training.
    numeric_df = df.select_dtypes(include='number')
    
    if target_column not in numeric_df.columns:
        # Throws error if the target column isn't numeric
        raise KeyError(f"'{target_column}' must be a numeric column in your CSV")
    
    X = numeric_df.drop(columns=[target_column])
    y = numeric_df[target_column]
    
    return X, y
