"""
Utility functions for the EPL prediction application
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import requests
import json

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.str_):
        return str(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def safe_json_serialize(obj):
    """Safely serialize objects to JSON, handling numpy types"""
    try:
        return json.dumps(convert_numpy_types(obj))
    except (TypeError, ValueError) as e:
        print(f"JSON serialization error: {e}")
        return json.dumps({"error": "Serialization failed"})

def load_models():
    """Load all trained models and their metadata"""
    
    models = {}
    
    try:
        # Load points prediction model
        if os.path.exists('models/points_model.joblib'):
            models['points'] = {
                'model': joblib.load('models/points_model.joblib'),
                'features': joblib.load('models/points_features.joblib')
            }
        
        # Load match result model
        if os.path.exists('models/match_result_model.joblib'):
            models['match_result'] = {
                'model': joblib.load('models/match_result_model.joblib'),
                'features': joblib.load('models/match_result_features.joblib')
            }
        
        # Load full features model
        if os.path.exists('models/full_features_model.joblib'):
            models['full_features'] = {
                'model': joblib.load('models/full_features_model.joblib'),
                'features': joblib.load('models/full_features_list.joblib'),
                'htr_encoder': joblib.load('models/htr_encoder.joblib')
            }
        
    except Exception as e:
        print(f"Error loading models: {e}")
    
    return models

def predict_match_result(match_data, model_type='full_features'):
    """Predict match result using trained models"""
    
    models = load_models()
    
    if model_type not in models:
        return None, None
    
    model_info = models[model_type]
    model = model_info['model']
    features = model_info['features']
    
    try:
        # Prepare input data
        if model_type == 'full_features' and 'HTR' in match_data:
            # Encode half-time result
            htr_encoder = model_info['htr_encoder']
            match_data['HTR_encoded'] = htr_encoder.transform([match_data['HTR']])[0]
        
        # Create feature vector
        X = np.array([[match_data.get(feat, 0) for feat in features]])
        
        # Make prediction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Convert numpy types to native Python types for JSON serialization
        prediction = convert_numpy_types(prediction)
        probabilities = convert_numpy_types(probabilities)
        
        # Create probability dictionary
        classes = model.classes_
        prob_dict = {convert_numpy_types(cls): convert_numpy_types(prob) 
                    for cls, prob in zip(classes, probabilities)}
        
        return prediction, prob_dict
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None

def get_team_stats():
    """Get team statistics from processed data"""
    
    try:
        if os.path.exists('outputs/team_statistics.csv'):
            return pd.read_csv('outputs/team_statistics.csv', index_col=0)
        else:
            # Return empty dataframe with expected columns
            return pd.DataFrame(columns=['Matches', 'Wins', 'Draws', 'Losses', 
                                       'GoalsFor', 'GoalsAgainst', 'Points'])
    except Exception as e:
        print(f"Error loading team stats: {e}")
        return pd.DataFrame()

def get_upcoming_matches():
    """Get upcoming EPL matches (mock data for demo)"""
    
    teams = ['Arsenal', 'Chelsea', 'Liverpool', 'Man City', 'Man United', 'Tottenham',
             'Newcastle', 'Brighton', 'Aston Villa', 'West Ham']
    
    import random
    from datetime import datetime, timedelta
    
    matches = []
    base_date = datetime.now()
    
    for i in range(10):
        match_date = base_date + timedelta(days=i+1)
        home_team = random.choice(teams)
        away_team = random.choice([t for t in teams if t != home_team])
        
        matches.append({
            'Date': match_date.strftime('%Y-%m-%d'),
            'Time': f"{random.randint(12, 20)}:{random.choice(['00', '30'])}",
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'Competition': 'Premier League'
        })
    
    return matches

def get_live_matches():
    """Get live EPL matches (mock data for demo)"""
    
    teams = ['Arsenal', 'Chelsea', 'Liverpool', 'Man City']
    
    import random
    
    matches = []
    
    # Generate 2-3 "live" matches
    for i in range(random.randint(2, 4)):
        home_team = random.choice(teams)
        away_team = random.choice([t for t in teams if t != home_team])
        
        matches.append({
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'HomeScore': random.randint(0, 3),
            'AwayScore': random.randint(0, 3),
            'Minute': random.randint(1, 90),
            'Status': 'Live'
        })
    
    return matches

def calculate_prediction_accuracy():
    """Calculate model accuracy metrics"""
    
    # Mock accuracy data for demo
    return {
        'half_time_accuracy': 68.5,
        'full_time_accuracy': 72.3,
        'total_predictions': 1247,
        'correct_predictions': 901
    }

def format_team_name(team_name):
    """Format team name for display"""
    
    # Handle common abbreviations
    abbreviations = {
        'Man City': 'Manchester City',
        'Man United': 'Manchester United',
        'Nottm Forest': 'Nottingham Forest',
        'Sheffield United': 'Sheffield Utd'
    }
    
    return abbreviations.get(team_name, team_name)

def get_team_colors():
    """Get team colors for visualization"""
    
    colors = {
        'Arsenal': '#EF0107',
        'Chelsea': '#034694',
        'Liverpool': '#C8102E',
        'Man City': '#6CABDD',
        'Manchester City': '#6CABDD',
        'Man United': '#DA020E',
        'Manchester United': '#DA020E',
        'Tottenham': '#132257',
        'Newcastle': '#241F20',
        'Brighton': '#0057B8',
        'Aston Villa': '#95BFE5',
        'West Ham': '#7A263A'
    }
    
    return colors

def validate_match_input(match_data):
    """Validate match input data"""
    
    required_fields = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
    
    errors = []
    
    for field in required_fields:
        if field not in match_data:
            errors.append(f"Missing field: {field}")
        else:
            try:
                value = float(match_data[field])
                if value < 0:
                    errors.append(f"{field} cannot be negative")
            except (ValueError, TypeError):
                errors.append(f"{field} must be a number")
    
    return errors

def get_feature_descriptions():
    """Get descriptions for all features"""
    
    descriptions = {
        'HS': 'Home Team Shots',
        'AS': 'Away Team Shots',
        'HST': 'Home Shots on Target',
        'AST': 'Away Shots on Target',
        'HC': 'Home Team Corners',
        'AC': 'Away Team Corners',
        'HF': 'Home Team Fouls',
        'AF': 'Away Team Fouls',
        'HY': 'Home Yellow Cards',
        'AY': 'Away Yellow Cards',
        'HR': 'Home Red Cards',
        'AR': 'Away Red Cards',
        'HTHG': 'Half Time Home Goals',
        'HTAG': 'Half Time Away Goals',
        'HTR': 'Half Time Result (H/D/A)'
    }
    
    return descriptions

def export_predictions(predictions, filename=None):
    """Export predictions to CSV"""
    
    if filename is None:
        filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    df = pd.DataFrame(predictions)
    filepath = os.path.join('outputs', filename)
    df.to_csv(filepath, index=False)
    
    return filepath