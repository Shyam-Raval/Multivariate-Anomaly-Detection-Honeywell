"""
Multivariate Time Series Anomaly Detection System

A comprehensive system for detecting anomalies in industrial sensor data
with feature attribution and explainable results.
"""

__version__ = "1.0.0"
__author__ = "Hackathon"

from .main import main, AnomalyDetectionSystem

__all__ = ["main", "AnomalyDetectionSystem"]