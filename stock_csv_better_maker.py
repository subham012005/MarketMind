import pandas as pd
def clean_multirow_csv(input_file, output_file):
    # Read the CSV file and skip the first 3 rows which are metadata
    df = pd.read_csv(input_file, skiprows=3, header=None)

    # Assign proper column names
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

    # Convert 'Date' to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Convert numeric columns to appropriate types
    numeric_cols = ['Close', 'High', 'Low', 'Open']
    df[numeric_cols] = df[numeric_cols].astype(float)
    df['Volume'] = df['Volume'].astype(int)

    # Save the cleaned CSV
    df.to_csv(output_file, index=False)
    print(f"Cleaned CSV saved to '{output_file}'")
