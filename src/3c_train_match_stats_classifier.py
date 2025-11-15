"""
Match Statistics Classifier Training
Specialized model for match outcome prediction using detailed statistics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def load_and_prepare_data():
    """Load and prepare data for match statistics classification"""
    
    # Load data
    if os.path.exists('data/epl_features.csv'):
        df = pd.read_csv('data/epl_features.csv')
    else:
        df = pd.read_csv('data/epl_combined.csv')
    
    # Select features for match statistics classification
    feature_columns = [
        'HS', 'AS',           # Shots
        'HST', 'AST',         # Shots on Target
        'HC', 'AC',           # Corners
        'HF', 'AF',           # Fouls
        'HY', 'AY',           # Yellow Cards
        'HR', 'AR'            # Red Cards
    ]
    
    # Prepare features and target
    X = df[feature_columns].fillna(0)
    y = df['FTR']  # Full Time Result
    
    # Remove any rows with missing target
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    
    return X, y, feature_columns

def train_multiple_models(X, y):
    """Train multiple models and compare performance"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize models
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=6),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Scale features for Logistic Regression
        if name == 'LogisticRegression':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Save scaler
            joblib.dump(scaler, f'models/{name.lower()}_scaler.joblib')
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
        
        print(f"{name} - Accuracy: {accuracy:.3f}")
        print(f"{name} - CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred))
    
    # Select best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.3f}")
    
    return best_model, best_model_name, results, X_test, y_test

def analyze_feature_importance(model, feature_columns):
    """Analyze and display feature importance"""
    
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(importance_df)
        
        # Save feature importance
        importance_df.to_csv('outputs/match_stats_feature_importance.csv', index=False)
        
        return importance_df
    else:
        print("Model does not support feature importance analysis")
        return None

def create_prediction_function(model, feature_columns, model_name):
    """Create a prediction function for the trained model"""
    
    def predict_match(match_stats):
        """
        Predict match outcome based on statistics
        
        Args:
            match_stats (dict): Dictionary with match statistics
            
        Returns:
            tuple: (prediction, probabilities)
        """
        try:
            # Prepare input
            input_data = np.array([[match_stats.get(col, 0) for col in feature_columns]])
            
            # Scale if needed
            if model_name == 'LogisticRegression':
                scaler = joblib.load('models/logisticregression_scaler.joblib')
                input_data = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            
            # Create probability dictionary
            classes = model.classes_
            prob_dict = {cls: prob for cls, prob in zip(classes, probabilities)}
            
            return prediction, prob_dict
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, None
    
    return predict_match

def save_model_artifacts(model, model_name, feature_columns, results):
    """Save all model artifacts"""
    
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_filename = f'models/match_stats_{model_name.lower()}.joblib'
    joblib.dump(model, model_filename)
    
    # Save feature columns
    joblib.dump(feature_columns, 'models/match_stats_features.joblib')
    
    # Save model metadata
    metadata = {
        'model_name': model_name,
        'model_type': 'match_statistics_classifier',
        'features': feature_columns,
        'accuracy': results[model_name]['accuracy'],
        'cv_score': results[model_name]['cv_mean'],
        'cv_std': results[model_name]['cv_std'],
        'training_date': pd.Timestamp.now().isoformat(),
        'classes': model.classes_.tolist()
    }
    
    import json
    with open('models/match_stats_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModel artifacts saved:")
    print(f"- Model: {model_filename}")
    print(f"- Features: models/match_stats_features.joblib")
    print(f"- Metadata: models/match_stats_metadata.json")

def main():
    """Main training function"""
    
    print("Starting Match Statistics Classifier Training...")
    
    # Load and prepare data
    X, y, feature_columns = load_and_prepare_data()
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Train models
    best_model, best_model_name, results, X_test, y_test = train_multiple_models(X, y)
    
    # Analyze feature importance
    importance_df = analyze_feature_importance(best_model, feature_columns)
    
    # Create prediction function
    predict_fn = create_prediction_function(best_model, feature_columns, best_model_name)
    
    # Test prediction function
    print("\nTesting prediction function...")
    sample_stats = {
        'HS': 15, 'AS': 8, 'HST': 6, 'AST': 3,
        'HC': 7, 'AC': 4, 'HF': 12, 'AF': 15,
        'HY': 2, 'AY': 3, 'HR': 0, 'AR': 0
    }
    
    pred, probs = predict_fn(sample_stats)
    if pred:
        print(f"Sample prediction: {pred}")
        print(f"Probabilities: {probs}")
    
    # Save model artifacts
    save_model_artifacts(best_model, best_model_name, feature_columns, results)
    
    print("\nMatch Statistics Classifier training complete!")

if __name__ == '__main__':
    main()