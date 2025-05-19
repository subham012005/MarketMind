import pandas as pd

def convert_stock_data(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Ensure the 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
    
    # Convert the 'Date' to only the date (remove time and timezone)
    df['Date'] = df['Date'].dt.date

    # Rearrange columns to match the desired order and remove Dividends and Stock Splits
    df = df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]  # Removed 'Dividends' and 'Stock Splits'

    # Save the cleaned data to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Converted data saved to '{output_file}'")
