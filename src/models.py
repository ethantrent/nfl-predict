# nfl-predict/src/models.py
"""
Machine learning models for NFL game outcome prediction.
Implements logistic regression and random forest classifiers.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def prepare_features(team_stats: pd.DataFrame, 
                    game_results: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare feature matrix for machine learning models.
    
    Args:
        team_stats: Aggregated team performance statistics
        game_results: Game outcomes (wins/losses)
        
    Returns:
        DataFrame with features and target variable
    """
    # Merge stats with results
    data = team_stats.merge(game_results, on=['game_id', 'team'], how='inner')
    
    # Select feature columns
    feature_cols = [
        'avg_epa', 'total_epa', 'avg_yards', 'total_yards',
        'success_rate', 'pass_epa', 'rush_epa', 
        'avg_pass_yards', 'avg_rush_yards', 'play_count'
    ]
    
    # Filter to available columns
    available_features = [col for col in feature_cols if col in data.columns]
    
    # Remove rows with missing values
    ml_data = data[available_features + ['won']].dropna()
    
    logger.info(f"Prepared {len(ml_data)} samples with {len(available_features)} features")
    
    return ml_data


def split_data(ml_data: pd.DataFrame, 
              test_size: float = 0.2,
              random_state: int = 42) -> Tuple:
    """
    Split data into training and testing sets.
    
    Args:
        ml_data: DataFrame with features and target
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler)
    """
    # Separate features and target
    X = ml_data.drop('won', axis=1)
    y = ml_data['won']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Testing set: {len(X_test)} samples")
    logger.info(f"Positive class ratio: {y_train.mean():.3f}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train.columns


def train_logistic_regression(X_train: np.ndarray, 
                              y_train: pd.Series,
                              random_state: int = 42) -> LogisticRegression:
    """
    Train logistic regression model for game outcome prediction.
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed
        
    Returns:
        Trained LogisticRegression model
    """
    logger.info("Training Logistic Regression model...")
    
    model = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    logger.info(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    return model


def train_random_forest(X_train: np.ndarray,
                       y_train: pd.Series,
                       random_state: int = 42) -> RandomForestClassifier:
    """
    Train random forest model for game outcome prediction.
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed
        
    Returns:
        Trained RandomForestClassifier model
    """
    logger.info("Training Random Forest model...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    logger.info(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: pd.Series,
                  model_name: str = "Model") -> Dict:
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for logging
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating {model_name}...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    logger.info(f"{model_name} Accuracy: {accuracy:.3f}")
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    return results


def get_feature_importance(model, feature_names: list,
                          top_n: int = 10) -> pd.DataFrame:
    """
    Extract and rank feature importance from trained model.
    
    Args:
        model: Trained model (LogisticRegression or RandomForest)
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        DataFrame with feature importance rankings
    """
    if hasattr(model, 'feature_importances_'):
        # Random Forest
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Logistic Regression
        importances = np.abs(model.coef_[0])
    else:
        logger.warning("Model does not support feature importance")
        return pd.DataFrame()
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    logger.info(f"Top {top_n} most important features:")
    for idx, row in importance_df.head(top_n).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return importance_df.head(top_n)


def predict_game_outcome(model, scaler, team_stats: Dict,
                         feature_names: list) -> Dict:
    """
    Predict outcome of a single game given team statistics.
    
    Args:
        model: Trained prediction model
        scaler: Fitted StandardScaler
        team_stats: Dictionary of team statistics
        feature_names: List of feature names in correct order
        
    Returns:
        Dictionary with prediction and probability
    """
    # Prepare feature vector
    features = [team_stats.get(feat, 0) for feat in feature_names]
    features_array = np.array(features).reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features_array)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    result = {
        'prediction': 'Win' if prediction == 1 else 'Loss',
        'win_probability': probability[1],
        'loss_probability': probability[0]
    }
    
    logger.info(f"Prediction: {result['prediction']} "
               f"(Win prob: {result['win_probability']:.3f})")
    
    return result


def compare_models(models_results: list) -> pd.DataFrame:
    """
    Compare performance of multiple models.
    
    Args:
        models_results: List of evaluation result dictionaries
        
    Returns:
        DataFrame with model comparison
    """
    comparison = []
    
    for result in models_results:
        comparison.append({
            'Model': result['model_name'],
            'Accuracy': result['accuracy'],
            'Precision': result['classification_report']['1']['precision'],
            'Recall': result['classification_report']['1']['recall'],
            'F1-Score': result['classification_report']['1']['f1-score']
        })
    
    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    
    logger.info("\nModel Comparison:")
    logger.info(comparison_df.to_string(index=False))
    
    return comparison_df

