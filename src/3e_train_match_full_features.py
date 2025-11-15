"""
Full Features Match Prediction Model
Advanced model using all available features including half-time data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_engineer_features():
    """Load data and create comprehensive feature set"""
    
    # Load data
    if os.path.exists('data/epl_features.csv'):
        df = pd.read_csv('data/epl_features.csv')
    else:
        df = pd.read_csv('data/epl_combined.csv')
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Base match statistics
    base_features = [
        'HS', 'AS', 'HST', 'AST', 'HC', 'AC',
        'HF', 'AF', 'HY', 'AY', 'HR', 'AR'
    ]
    
    # Half-time features
    halftime_features = ['HTHG', 'HTAG']
    
    # Create additional engineered features
    df['TotalShots'] = df['HS'] + df['AS']
    df['ShotDifference'] = df['HS'] - df['AS']
    df['ShotsOnTargetRatio'] = (df['HST'] + df['AST']) / (df['HS'] + df['AS'] + 1)
    df['HomeShotAccuracy'] = df['HST'] / (df['HS'] + 1)
    df['AwayShotAccuracy'] = df['AST'] / (df['AS'] + 1)
    df['ShotAccuracyDiff'] = df['HomeShotAccuracy'] - df['AwayShotAccuracy']
    
    df['TotalCorners'] = df['HC'] + df['AC']
    df['CornerDifference'] = df['HC'] - df['AC']
    
    df['TotalFouls'] = df['HF'] + df['AF']
    df['FoulDifference'] = df['HF'] - df['AF']
    
    df['TotalCards'] = df['HY'] + df['AY'] + df['HR'] + df['AR']
    df['HomeCards'] = df['HY'] + df['HR']
    df['AwayCards'] = df['AY'] + df['AR']
    df['CardDifference'] = df['HomeCards'] - df['AwayCards']
    
    # Half-time features
    df['HTGoalDifference'] = df['HTHG'] - df['HTAG']
    df['HTTotalGoals'] = df['HTHG'] + df['HTAG']
    
    # Encode half-time result
    le_htr = LabelEncoder()
    df['HTR_encoded'] = le_htr.fit_transform(df['HTR'].fillna('D'))
    
    # Time-based features
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    
    # All features for the model
    feature_columns = (
        base_features + 
        halftime_features + 
        ['TotalShots', 'ShotDifference', 'ShotsOnTargetRatio', 
         'HomeShotAccuracy', 'AwayShotAccuracy', 'ShotAccuracyDiff',
         'TotalCorners', 'CornerDifference', 'TotalFouls', 'FoulDifference',
         'TotalCards', 'HomeCards', 'AwayCards', 'CardDifference',
         'HTGoalDifference', 'HTTotalGoals', 'HTR_encoded',
         'Month', 'DayOfWeek', 'IsWeekend']
    )
    
    return df, feature_columns, le_htr

def select_best_features(X, y, feature_columns, k=20):
    """Select the best features using statistical tests"""
    
    print(f"Selecting top {k} features from {len(feature_columns)} available features...")
    
    # Use SelectKBest with f_classif
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = [feature_columns[i] for i in range(len(feature_columns)) if selected_mask[i]]
    
    # Get feature scores
    feature_scores = pd.DataFrame({
        'feature': feature_columns,
        'score': selector.scores_,
        'selected': selected_mask
    }).sort_values('score', ascending=False)
    
    print("Top 10 features by score:")
    print(feature_scores.head(10))
    
    # Save feature selection results
    feature_scores.to_csv('outputs/full_features_selection.csv', index=False)
    
    return X_selected, selected_features, selector

def optimize_hyperparameters(X_train, y_train):
    """Optimize model hyperparameters using GridSearchCV"""
    
    print("Optimizing hyperparameters...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    
    # Initialize model
    rf = RandomForestClassifier(random_state=42)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='accuracy', 
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.3f}")
    
    return grid_search.best_estimator_

def train_ensemble_model(X_train, y_train, X_test, y_test):
    """Train an ensemble of models"""
    
    print("Training ensemble model...")
    
    # Individual models
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    et = ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=42)
    
    # Train models
    rf.fit(X_train, y_train)
    et.fit(X_train, y_train)
    
    # Make predictions
    rf_pred = rf.predict_proba(X_test)
    et_pred = et.predict_proba(X_test)
    
    # Ensemble prediction (average probabilities)
    ensemble_pred_proba = (rf_pred + et_pred) / 2
    ensemble_pred = rf.classes_[np.argmax(ensemble_pred_proba, axis=1)]
    
    # Evaluate ensemble
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    
    print(f"Random Forest accuracy: {accuracy_score(y_test, rf.predict(X_test)):.3f}")
    print(f"Extra Trees accuracy: {accuracy_score(y_test, et.predict(X_test)):.3f}")
    print(f"Ensemble accuracy: {ensemble_accuracy:.3f}")
    
    # Return the best individual model for simplicity
    return rf if accuracy_score(y_test, rf.predict(X_test)) > accuracy_score(y_test, et.predict(X_test)) else et

def analyze_model_performance(model, X_test, y_test, selected_features):
    """Comprehensive model performance analysis"""
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.3f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix - Full Features Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('outputs/full_features_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': selected_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'], color='#9c88ff')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Feature Importances - Full Features Model')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('outputs/full_features_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save importance
        importance_df.to_csv('outputs/full_features_importance.csv', index=False)
        
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10))
    
    return accuracy, y_pred, y_pred_proba

def save_full_model(model, selected_features, feature_selector, label_encoder, accuracy):
    """Save the complete model pipeline"""
    
    os.makedirs('models', exist_ok=True)
    
    # Save model components
    joblib.dump(model, 'models/full_features_model.joblib')
    joblib.dump(selected_features, 'models/full_features_list.joblib')
    joblib.dump(feature_selector, 'models/full_features_selector.joblib')
    joblib.dump(label_encoder, 'models/full_htr_encoder.joblib')
    
    # Save metadata
    metadata = {
        'model_type': 'full_features_classifier',
        'model_class': str(type(model).__name__),
        'n_features': len(selected_features),
        'features': selected_features,
        'accuracy': accuracy,
        'training_date': pd.Timestamp.now().isoformat(),
        'classes': model.classes_.tolist()
    }
    
    import json
    with open('models/full_features_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nFull features model saved with {len(selected_features)} features")
    print(f"Test accuracy: {accuracy:.3f}")

def create_prediction_pipeline():
    """Create a complete prediction pipeline"""
    
    def predict_match_full_features(match_data):
        """
        Predict match outcome using full features
        
        Args:
            match_data (dict): Complete match data including half-time stats
            
        Returns:
            tuple: (prediction, probabilities, feature_contributions)
        """
        try:
            # Load model components
            model = joblib.load('models/full_features_model.joblib')
            selected_features = joblib.load('models/full_features_list.joblib')
            feature_selector = joblib.load('models/full_features_selector.joblib')
            htr_encoder = joblib.load('models/full_htr_encoder.joblib')
            
            # Engineer features from input data
            engineered_data = engineer_prediction_features(match_data, htr_encoder)
            
            # Select features
            feature_vector = np.array([[engineered_data.get(feat, 0) for feat in selected_features]])
            
            # Make prediction
            prediction = model.predict(feature_vector)[0]
            probabilities = model.predict_proba(feature_vector)[0]
            
            # Create probability dictionary
            prob_dict = {cls: prob for cls, prob in zip(model.classes_, probabilities)}
            
            # Feature contributions (if available)
            contributions = None
            if hasattr(model, 'feature_importances_'):
                contributions = {
                    feat: imp * engineered_data.get(feat, 0) 
                    for feat, imp in zip(selected_features, model.feature_importances_)
                }
            
            return prediction, prob_dict, contributions
            
        except Exception as e:
            print(f"Full features prediction error: {e}")
            return None, None, None
    
    return predict_match_full_features

def engineer_prediction_features(match_data, htr_encoder):
    """Engineer features for prediction from raw match data"""
    
    engineered = match_data.copy()
    
    # Calculate derived features
    engineered['TotalShots'] = match_data.get('HS', 0) + match_data.get('AS', 0)
    engineered['ShotDifference'] = match_data.get('HS', 0) - match_data.get('AS', 0)
    
    total_shots = engineered['TotalShots']
    engineered['ShotsOnTargetRatio'] = (
        (match_data.get('HST', 0) + match_data.get('AST', 0)) / (total_shots + 1)
    )
    
    engineered['HomeShotAccuracy'] = match_data.get('HST', 0) / (match_data.get('HS', 0) + 1)
    engineered['AwayShotAccuracy'] = match_data.get('AST', 0) / (match_data.get('AS', 0) + 1)
    engineered['ShotAccuracyDiff'] = engineered['HomeShotAccuracy'] - engineered['AwayShotAccuracy']
    
    engineered['TotalCorners'] = match_data.get('HC', 0) + match_data.get('AC', 0)
    engineered['CornerDifference'] = match_data.get('HC', 0) - match_data.get('AC', 0)
    
    engineered['TotalFouls'] = match_data.get('HF', 0) + match_data.get('AF', 0)
    engineered['FoulDifference'] = match_data.get('HF', 0) - match_data.get('AF', 0)
    
    home_cards = match_data.get('HY', 0) + match_data.get('HR', 0)
    away_cards = match_data.get('AY', 0) + match_data.get('AR', 0)
    engineered['TotalCards'] = home_cards + away_cards
    engineered['HomeCards'] = home_cards
    engineered['AwayCards'] = away_cards
    engineered['CardDifference'] = home_cards - away_cards
    
    # Half-time features
    engineered['HTGoalDifference'] = match_data.get('HTHG', 0) - match_data.get('HTAG', 0)
    engineered['HTTotalGoals'] = match_data.get('HTHG', 0) + match_data.get('HTAG', 0)
    
    # Encode HTR if provided
    if 'HTR' in match_data:
        try:
            engineered['HTR_encoded'] = htr_encoder.transform([match_data['HTR']])[0]
        except:
            engineered['HTR_encoded'] = htr_encoder.transform(['D'])[0]  # Default to Draw
    else:
        engineered['HTR_encoded'] = htr_encoder.transform(['D'])[0]
    
    # Time features (use defaults if not provided)
    engineered['Month'] = match_data.get('Month', 9)  # Default to September
    engineered['DayOfWeek'] = match_data.get('DayOfWeek', 5)  # Default to Saturday
    engineered['IsWeekend'] = 1 if engineered['DayOfWeek'] >= 5 else 0
    
    return engineered

def main():
    """Main training function"""
    
    print("Starting Full Features Model Training...")
    
    # Load and engineer features
    df, feature_columns, le_htr = load_and_engineer_features()
    print(f"Data loaded: {df.shape[0]} samples")
    print(f"Total features available: {len(feature_columns)}")
    
    # Prepare data
    X = df[feature_columns].fillna(0)
    y = df['FTR']
    
    # Remove missing targets
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    
    print(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Feature selection
    X_selected, selected_features, selector = select_best_features(X, y, feature_columns, k=20)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train optimized model
    model = optimize_hyperparameters(X_train, y_train)
    
    # Train ensemble (optional)
    # model = train_ensemble_model(X_train, y_train, X_test, y_test)
    
    # Analyze performance
    accuracy, y_pred, y_pred_proba = analyze_model_performance(
        model, X_test, y_test, selected_features
    )
    
    # Save model
    save_full_model(model, selected_features, selector, le_htr, accuracy)
    
    # Create prediction pipeline
    predict_fn = create_prediction_pipeline()
    
    print("\nFull Features Model training complete!")
    print(f"Final model accuracy: {accuracy:.3f}")

if __name__ == '__main__':
    main()