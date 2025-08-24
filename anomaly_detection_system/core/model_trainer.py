"""
Model training module for anomaly detection using Isolation Forest.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, Optional
from sklearn.ensemble import IsolationForest

from ..utils.config import SystemConfig


class ModelTrainer:
    """
    Handles training and validation of the Isolation Forest anomaly detection model.
    
    This class is responsible for:
    - Training Isolation Forest model on normal period data
    - Validating training performance against thresholds
    - Managing model parameters and configuration
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize ModelTrainer with system configuration.
        
        Args:
            config: System configuration containing model parameters
        """
        self.config = config
        self.model: Optional[IsolationForest] = None
        self.training_scores: Optional[np.ndarray] = None
        self.is_trained = False
    
    def train_model(self, training_data: pd.DataFrame) -> IsolationForest:
        """
        Train Isolation Forest model on the provided training data.
        
        Args:
            training_data: Clean numerical training data
            
        Returns:
            Trained IsolationForest model
            
        Raises:
            ValueError: If training data is invalid or training fails
        """
        if training_data.empty:
            raise ValueError("Training data cannot be empty")
        
        if training_data.isnull().any().any():
            raise ValueError("Training data contains missing values")
        
        print(f"Training Isolation Forest model on {len(training_data)} samples...")
        print(f"Features: {training_data.shape[1]}")
        
        # Initialize model with configured parameters
        self.model = IsolationForest(**self.config.model_params)
        
        try:
            # Train the model
            self.model.fit(training_data)
            self.is_trained = True
            
            print("Model training completed successfully")
            
            # Calculate training scores for validation
            self.training_scores = self.model.decision_function(training_data)
            
            return self.model
            
        except Exception as e:
            raise ValueError(f"Model training failed: {e}")
    
    def validate_training_scores(self, model: IsolationForest, data: pd.DataFrame) -> bool:
        """
        Validate that training scores meet quality thresholds.
        
        Args:
            model: Trained Isolation Forest model
            data: Training data used for validation
            
        Returns:
            True if validation passes, False otherwise
        """
        if not self.is_trained or model is None:
            raise ValueError("Model must be trained before validation")
        
        # Get decision function scores (more negative = more anomalous)
        scores = model.decision_function(data)
        
        # Transform to positive scale for validation (higher = more anomalous)
        # Note: We'll use a simple transformation for validation purposes
        validation_scores = -scores  # Flip sign so higher = more anomalous
        validation_scores = (validation_scores - validation_scores.min()) * 100 / (validation_scores.max() - validation_scores.min())
        
        mean_score = validation_scores.mean()
        max_score = validation_scores.max()
        
        print(f"Training validation scores - Mean: {mean_score:.2f}, Max: {max_score:.2f}")
        
        # Check against thresholds
        mean_threshold = self.config.training_score_mean_threshold
        max_threshold = self.config.training_score_max_threshold
        
        mean_valid = mean_score < mean_threshold
        max_valid = max_score < max_threshold
        
        if not mean_valid:
            print(f"INFO: Training mean score ({mean_score:.2f}) exceeds threshold ({mean_threshold}). "
                  "This is normal for industrial data with natural variations.")
        
        if not max_valid:
            print(f"INFO: Training max score ({max_score:.2f}) exceeds threshold ({max_threshold}). "
                  "This indicates some natural anomalies in the training period, which is expected.")
        
        validation_passed = mean_valid and max_valid
        
        if validation_passed:
            print("Training validation passed - scores within acceptable ranges")
        else:
            print("Training validation failed - proceeding with caution")
        
        return validation_passed
    
    def get_model_parameters(self) -> Dict:
        """
        Get the current model parameters and training statistics.
        
        Returns:
            Dictionary containing model parameters and training info
        """
        if not self.is_trained or self.model is None:
            return {"status": "not_trained"}
        
        params = {
            "status": "trained",
            "model_type": "IsolationForest",
            "n_estimators": self.model.n_estimators,
            "contamination": self.model.contamination,
            "random_state": self.model.random_state,
            "max_samples": self.model.max_samples,
            "max_features": self.model.max_features
        }
        
        if self.training_scores is not None:
            params.update({
                "training_score_mean": float(self.training_scores.mean()),
                "training_score_std": float(self.training_scores.std()),
                "training_score_min": float(self.training_scores.min()),
                "training_score_max": float(self.training_scores.max())
            })
        
        return params
    
    def get_model(self) -> IsolationForest:
        """
        Get the trained model instance.
        
        Returns:
            Trained IsolationForest model
            
        Raises:
            ValueError: If model hasn't been trained yet
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before use")
        
        return self.model