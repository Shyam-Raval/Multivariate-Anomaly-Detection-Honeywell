"""
Anomaly detection module for applying trained models and generating scores.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Tuple
from sklearn.ensemble import IsolationForest
from scipy import stats

from ..utils.config import SystemConfig


class AnomalyDetector:
    """
    Handles anomaly detection and scoring using trained models.
    
    This class is responsible for:
    - Applying trained models to detect anomalies
    - Calculating and transforming anomaly scores to 0-100 scale
    - Handling edge cases and score normalization
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize AnomalyDetector with system configuration.
        
        Args:
            config: System configuration containing scoring parameters
        """
        self.config = config
    
    def detect_anomalies(self, model: IsolationForest, data: pd.DataFrame) -> np.ndarray:
        """
        Detect anomalies using the trained model.
        
        Args:
            model: Trained IsolationForest model
            data: Data to analyze for anomalies
            
        Returns:
            Binary anomaly predictions (1 for normal, -1 for anomaly)
            
        Raises:
            ValueError: If model or data is invalid
        """
        if data.empty:
            raise ValueError("Analysis data cannot be empty")
        
        if data.isnull().any().any():
            raise ValueError("Analysis data contains missing values")
        
        print(f"Detecting anomalies in {len(data)} samples...")
        
        try:
            # Get binary predictions (1 for normal, -1 for anomaly)
            predictions = model.predict(data)
            
            anomaly_count = np.sum(predictions == -1)
            normal_count = np.sum(predictions == 1)
            
            print(f"Anomaly detection complete: {normal_count} normal, {anomaly_count} anomalous")
            
            return predictions
            
        except Exception as e:
            raise ValueError(f"Anomaly detection failed: {e}")
    
    def calculate_abnormality_scores(self, model: IsolationForest, data: pd.DataFrame) -> np.ndarray:
        """
        Calculate raw abnormality scores using model decision function.
        
        Args:
            model: Trained IsolationForest model
            data: Data to score
            
        Returns:
            Raw anomaly scores from decision function
        """
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        try:
            # Get decision function scores (more negative = more anomalous)
            raw_scores = model.decision_function(data)
            
            print(f"Raw scores - Min: {raw_scores.min():.4f}, Max: {raw_scores.max():.4f}, Mean: {raw_scores.mean():.4f}")
            
            return raw_scores
            
        except Exception as e:
            raise ValueError(f"Score calculation failed: {e}")
    
    def transform_to_percentile_scale(self, scores: np.ndarray) -> np.ndarray:
        """
        Transform raw scores to 0-100 percentile scale.
        
        Args:
            scores: Raw anomaly scores from decision function
            
        Returns:
            Transformed scores on 0-100 scale
        """
        if len(scores) == 0:
            raise ValueError("Scores array cannot be empty")
        
        # Handle edge case where all scores are identical
        if np.all(scores == scores[0]):
            warnings.warn("All scores are identical, adding small noise to avoid zero variance")
            scores = scores + np.random.normal(0, self.config.score_noise_factor, len(scores))
        
        # Transform to percentile ranking (0-100 scale)
        # More negative decision function scores = higher percentile = higher abnormality
        percentile_scores = stats.rankdata(-scores, method='average') / len(scores) * 100
        
        # Ensure scores are in valid range
        percentile_scores = np.clip(percentile_scores, 0.0, 100.0)
        
        # Add small noise to avoid exactly 0 scores if needed
        zero_mask = percentile_scores == 0.0
        if np.any(zero_mask):
            percentile_scores[zero_mask] += np.random.uniform(
                self.config.score_noise_factor, 
                self.config.score_noise_factor * 10, 
                np.sum(zero_mask)
            )
        
        print(f"Transformed scores - Min: {percentile_scores.min():.2f}, Max: {percentile_scores.max():.2f}, Mean: {percentile_scores.mean():.2f}")
        
        return percentile_scores
    
    def validate_score_distribution(self, scores: np.ndarray, training_data_size: int) -> bool:
        """
        Validate that score distribution meets expected criteria.
        
        Args:
            scores: Transformed abnormality scores (0-100 scale)
            training_data_size: Size of training dataset for context
            
        Returns:
            True if distribution is valid, False otherwise
        """
        mean_score = scores.mean()
        std_score = scores.std()
        min_score = scores.min()
        max_score = scores.max()
        
        print(f"Score distribution validation:")
        print(f"  Mean: {mean_score:.2f}, Std: {std_score:.2f}")
        print(f"  Min: {min_score:.2f}, Max: {max_score:.2f}")
        
        # Check for reasonable distribution
        issues = []
        
        # Check if scores are too concentrated
        if std_score < 5.0:
            issues.append("Low score variance - may indicate poor model discrimination")
        
        # Check for extreme skewness
        if mean_score > 80.0:
            issues.append("Very high mean score - most data appears anomalous")
        elif mean_score < 5.0:
            issues.append("Very low mean score - most data appears normal")
        
        # Check range utilization
        score_range = max_score - min_score
        if score_range < 20.0:
            issues.append("Limited score range - poor anomaly discrimination")
        
        if issues:
            for issue in issues:
                warnings.warn(f"Score distribution issue: {issue}")
            return False
        
        print("Score distribution validation passed")
        return True
    
    def process_analysis_data(self, model: IsolationForest, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete anomaly detection and scoring pipeline.
        
        Args:
            model: Trained IsolationForest model
            data: Analysis data to process
            
        Returns:
            Tuple of (binary_predictions, abnormality_scores)
        """
        print("Starting anomaly detection and scoring pipeline...")
        
        # Get binary anomaly predictions
        binary_predictions = self.detect_anomalies(model, data)
        
        # Calculate raw scores
        raw_scores = self.calculate_abnormality_scores(model, data)
        
        # Transform to 0-100 scale
        abnormality_scores = self.transform_to_percentile_scale(raw_scores)
        
        # Validate score distribution
        self.validate_score_distribution(abnormality_scores, len(data))
        
        print("Anomaly detection and scoring pipeline completed")
        
        return binary_predictions, abnormality_scores