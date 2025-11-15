"""
Model Training Script
Trains multiple ML models for EPL predictions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
import joblib
import os

def prepare_data():
    """Load and prepare data for training"""
    
    # Load data
    if os.path.exists('data/epl_features.csv'):
        df = pd.read_csv('data/epl_features.csv')
    else:
        df = pd.read_csv('data/epl_combined.csv')
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Create team statistics for points prediction
    team_stats = calculate_team_features(df)
    
    return df, team_stats

def calculate_team_features(df):
    """Calculate team-level features for points prediction"""
    
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    team_features = []
    
    for team in teams:
        home_matches = df[df['HomeTeam'] == team]
        away_matches = df[df['AwayTeam'] == team]
        
        # Basic stats
        total_matches = len(home_matches) + len(away_matches)
        if total_matches == 0:
            continue
            
        # Goals
        goals_for = home_matches['FTHG'].sum() + away_matches['FTAG'].sum()
        goals_against = home_matches['FTAG'].sum() + away_matches['FTHG'].sum()
        
        # Shots
        shots_for = home_matches['HS'].sum() + away_matches['AS'].sum()
        shots_against = home_matches['AS'].sum() + away_matches['HS'].sum()
        shots_on_target_for = home_matches['HST'].sum() + away_matches['AST'].sum()
        shots_on_target_against = home_matches['AST'].sum() + away_matches['HST'].sum()
        
        # Cards and fouls
        yellow_cards = home_matches['HY'].sum() + away_matches['AY'].sum()
        red_cards = home_matches['HR'].sum() + away_matches['AR'].sum()
        fouls = home_matches['HF'].sum() + away_matches['AF'].sum()
        
        # Corners
        corners_for = home_matches['HC'].sum() + away_matches['AC'].sum()
        corners_against = home_matches['AC'].sum() + away_matches['HC'].sum()
        
        # Results
        home_wins = len(home_matches[home_matches['FTR'] == 'H'])
        away_wins = len(away_matches[away_matches['FTR'] == 'A'])
        home_draws = len(home_matches[home_matches['FTR'] == 'D'])
        away_draws = len(away_matches[away_matches['FTR'] == 'D'])
        
        total_wins = home_wins + away_wins
        total_draws = home_draws + away_draws
        points = total_wins * 3 + total_draws
        
        # Calculate averages
        team_features.append({
            'Team': team,
            'AvgGoalsFor': goals_for / total_matches,
            'AvgGoalsAgainst': goals_against / total_matches,
            'AvgShotsFor': shots_for / total_matches,
            'AvgShotsAgainst': shots_against / total_matches,
            'AvgSOTFor': shots_on_target_for / total_matches,
            'AvgSOTAgainst': shots_on_target_against / total_matches,
            'AvgYellowCards': yellow_cards / total_matches,
            'AvgRedCards': red_cards / total_matches,
            'AvgFouls': fouls / total_matches,
            'AvgCornersFor': corners_for / total_matches,
            'AvgCornersAgainst': corners_against / total_matches,
            'WinRate': total_wins / total_matches,
            'TotalPoints': points
        })
    
    return pd.DataFrame(team_features)

def train_points_model(team_stats):
    """Train model to predict total points"""
    
    print("Training points prediction model...")
    
    # Features for points prediction
    feature_cols = ['AvgGoalsFor', 'AvgGoalsAgainst', 'AvgShotsFor', 'AvgShotsAgainst',
                   'AvgSOTFor', 'AvgSOTAgainst', 'AvgYellowCards', 'AvgRedCards',
                   'AvgFouls', 'AvgCornersFor', 'AvgCornersAgainst', 'WinRate']
    
    X = team_stats[feature_cols]
    y = team_stats['TotalPoints']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Points Model - MAE: {mae:.2f}, R²: {r2:.3f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/points_model.joblib')
    joblib.dump(feature_cols, 'models/points_features.joblib')
    
    return model, feature_cols

def train_match_result_model(df):
    """Train model to predict match results"""
    
    print("Training match result prediction model...")
    
    # Basic features for match prediction
    feature_cols = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
    
    # Prepare data
    X = df[feature_cols].fillna(0)
    y = df['FTR']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Match Result Model - Accuracy: {accuracy:.3f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(model, 'models/match_result_model.joblib')
    joblib.dump(feature_cols, 'models/match_result_features.joblib')
    
    return model, feature_cols

def train_full_features_model(df):
    """Train model with full features including half-time data"""
    
    print("Training full features model...")
    
    # Full feature set
    feature_cols = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 
                   'HY', 'AY', 'HR', 'AR', 'HTHG', 'HTAG']
    
    # Add half-time result encoding
    df_model = df.copy()
    le_htr = LabelEncoder()
    df_model['HTR_encoded'] = le_htr.fit_transform(df_model['HTR'].fillna('D'))
    feature_cols.append('HTR_encoded')
    
    # Prepare data
    X = df_model[feature_cols].fillna(0)
    y = df_model['FTR']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Full Features Model - Accuracy: {accuracy:.3f}")
    
    # Save model and encoders
    joblib.dump(model, 'models/full_features_model.joblib')
    joblib.dump(feature_cols, 'models/full_features_list.joblib')
    joblib.dump(le_htr, 'models/htr_encoder.joblib')
    
    return model, feature_cols

def main():
    """Main training function"""
    
    print("Starting model training...")
    
    # Prepare data
    df, team_stats = prepare_data()
    
    # Train all models
    points_model, points_features = train_points_model(team_stats)
    match_model, match_features = train_match_result_model(df)
    full_model, full_features = train_full_features_model(df)
    
    # Save metadata
    metadata = {
        'models_trained': ['points_model', 'match_result_model', 'full_features_model'],
        'training_date': pd.Timestamp.now().isoformat(),
        'data_shape': df.shape,
        'team_count': len(team_stats)
    }
    
    import json
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("All models trained and saved successfully!")
    print(f"Models saved in 'models/' directory")

if __name__ == '__main__':
    main()