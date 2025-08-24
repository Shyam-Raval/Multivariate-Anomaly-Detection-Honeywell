"""
Configuration management for the anomaly detection system.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class SystemConfig:
    """
    Configuration parameters for the anomaly detection system.
    
    Attributes:
        model_params: Parameters for the Isolation Forest model
        training_period: Start and end timestamps for training period
        analysis_period: Start and end timestamps for analysis period  
        shap_background_size: Number of samples for SHAP background
        max_runtime_minutes: Maximum allowed runtime in minutes
        score_noise_factor: Small noise factor to avoid exactly 0 scores
    """
    model_params: Dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'contamination': 0.05,
        'random_state': 42
    })
    
    training_period: Tuple[str, str] = (
        '2004-01-01 00:00:00', 
        '2004-01-05 23:59:00'
    )
    
    analysis_period: Tuple[str, str] = (
        '2004-01-01 00:00:00', 
        '2004-01-19 07:59:00'
    )
    
    shap_background_size: int = 100
    max_runtime_minutes: int = 15
    score_noise_factor: float = 1e-6
    
    # Validation thresholds
    training_score_mean_threshold: float = 10.0
    training_score_max_threshold: float = 25.0
    normal_score_range: Tuple[float, float] = (0.0, 20.0)
    
    # Feature attribution settings
    min_feature_contribution_pct: float = 1.0
    max_top_features: int = 7