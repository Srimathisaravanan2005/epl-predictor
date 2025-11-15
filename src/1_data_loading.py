"""
Data Loading and Preprocessing Script
Loads EPL data from multiple CSV files and combines them into a single dataset
"""

import pandas as pd
import os
import glob
from datetime import datetime

def load_and_combine_data():
    """Load and combine EPL data from multiple CSV files"""
    
    # Required columns to retain
    required_columns = [
        'Date', 'HomeTeam', 'AwayTeam',
        'HTHG', 'HTAG', 'HTR',
        'FTHG', 'FTAG', 'FTR',
        'HS', 'AS', 'HST', 'AST', 'HC', 'AC',
        'HF', 'AF', 'HY', 'AY', 'HR', 'AR'
    ]
    
    # Get all CSV files from data directory
    data_dir = 'data'
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    if not csv_files:
        print("No CSV files found in data directory. Creating sample data...")
        create_sample_data()
        csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    combined_data = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            print(f"Loading {file}: {len(df)} rows")
            
            # Keep only required columns that exist in the file
            available_cols = [col for col in required_columns if col in df.columns]
            df = df[available_cols]
            
            combined_data.append(df)
            
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not combined_data:
        print("No valid data found. Creating sample data...")
        create_sample_data()
        return load_and_combine_data()
    
    # Combine all dataframes
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Clean and process data
    combined_df = clean_data(combined_df)
    
    # Save combined data
    output_path = os.path.join(data_dir, 'epl_combined.csv')
    combined_df.to_csv(output_path, index=False)
    print(f"Combined data saved to {output_path}: {len(combined_df)} rows, {len(combined_df.columns)} columns")
    
    return combined_df

def clean_data(df):
    """Clean and preprocess the combined dataset"""
    
    # Parse dates
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Fill missing values with appropriate defaults
    numeric_columns = ['HTHG', 'HTAG', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 
                      'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Fill categorical columns
    categorical_columns = ['HTR', 'FTR']
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna('D')  # Default to Draw
    
    # Remove rows with missing essential data
    essential_cols = ['HomeTeam', 'AwayTeam', 'FTR']
    for col in essential_cols:
        if col in df.columns:
            df = df.dropna(subset=[col])
    
    return df

def create_sample_data():
    """Create sample EPL data for demonstration"""
    
    teams = ['Arsenal', 'Chelsea', 'Liverpool', 'Man City', 'Man United', 'Tottenham',
             'Newcastle', 'Brighton', 'Aston Villa', 'West Ham', 'Crystal Palace', 
             'Fulham', 'Brentford', 'Wolves', 'Everton', 'Nottm Forest', 
             'Bournemouth', 'Luton', 'Burnley', 'Sheffield United']
    
    import random
    import numpy as np
    
    # Generate sample matches
    matches = []
    for season in range(2020, 2024):
        for home_team in teams:
            for away_team in teams:
                if home_team != away_team:
                    # Generate realistic match data
                    hthg = random.randint(0, 3)
                    htag = random.randint(0, 2)
                    fthg = hthg + random.randint(0, 2)
                    ftag = htag + random.randint(0, 2)
                    
                    # Determine results
                    if hthg > htag:
                        htr = 'H'
                    elif hthg < htag:
                        htr = 'A'
                    else:
                        htr = 'D'
                    
                    if fthg > ftag:
                        ftr = 'H'
                    elif fthg < ftag:
                        ftr = 'A'
                    else:
                        ftr = 'D'
                    
                    match = {
                        'Date': f"{season}-{random.randint(8, 12):02d}-{random.randint(1, 28):02d}",
                        'HomeTeam': home_team,
                        'AwayTeam': away_team,
                        'HTHG': hthg,
                        'HTAG': htag,
                        'HTR': htr,
                        'FTHG': fthg,
                        'FTAG': ftag,
                        'FTR': ftr,
                        'HS': random.randint(8, 20),
                        'AS': random.randint(6, 18),
                        'HST': random.randint(2, 8),
                        'AST': random.randint(1, 7),
                        'HC': random.randint(2, 12),
                        'AC': random.randint(1, 10),
                        'HF': random.randint(8, 20),
                        'AF': random.randint(8, 20),
                        'HY': random.randint(0, 5),
                        'AY': random.randint(0, 5),
                        'HR': random.randint(0, 1),
                        'AR': random.randint(0, 1)
                    }
                    matches.append(match)
    
    # Create sample datasets
    sample_df = pd.DataFrame(matches)
    
    # Split into multiple files to simulate real data
    seasons = sample_df['Date'].str[:4].unique()
    
    os.makedirs('data', exist_ok=True)
    
    for season in seasons:
        season_data = sample_df[sample_df['Date'].str.startswith(season)]
        season_data.to_csv(f'data/epl_{season}.csv', index=False)
        print(f"Created sample data for season {season}: {len(season_data)} matches")

if __name__ == '__main__':
    print("Starting data loading and preprocessing...")
    df = load_and_combine_data()
    print(f"Data processing complete. Final dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")