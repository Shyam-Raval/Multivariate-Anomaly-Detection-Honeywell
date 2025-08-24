"""
Output generation module for creating formatted CSV results.
"""

import pandas as pd
import numpy as np
import os
from typing import List, Optional
from pathlib import Path

from ..utils.config import SystemConfig
from ..utils.validators import validate_output_path


class OutputGenerator:
    """
    Handles output generation and CSV formatting for anomaly detection results.
    
    This class is responsible for:
    - Combining original data with anomaly detection results
    - Formatting output according to hackathon specifications
    - Validating output format and saving to CSV
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize OutputGenerator with system configuration.
        
        Args:
            config: System configuration containing output parameters
        """
        self.config = config
    
    def create_output_dataframe(self, original_data: pd.DataFrame,
                              analysis_data: pd.DataFrame,
                              abnormality_scores: np.ndarray,
                              top_features_df: pd.DataFrame,
                              binary_predictions: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Create the final output DataFrame with all required columns.
        
        Args:
            original_data: Original CSV data with all columns
            analysis_data: Processed analysis data (subset of original)
            abnormality_scores: Anomaly scores (0-100 scale)
            top_features_df: DataFrame with top contributing features
            binary_predictions: Optional binary anomaly predictions
            
        Returns:
            Complete output DataFrame ready for CSV export
            
        Raises:
            ValueError: If data dimensions don't match or required columns missing
        """
        print("Creating output DataFrame...")
        
        # Validate input dimensions
        if len(abnormality_scores) != len(analysis_data):
            raise ValueError("Abnormality scores length doesn't match analysis data")
        
        if len(top_features_df) != len(analysis_data):
            raise ValueError("Top features DataFrame length doesn't match analysis data")
        
        # Start with the analysis period from original data
        # Get the time range that matches analysis_data
        analysis_start_time = analysis_data.index[0]
        analysis_end_time = analysis_data.index[-1]
        
        # Filter original data to analysis period
        output_df = original_data.loc[analysis_start_time:analysis_end_time].copy()
        
        print(f"Output DataFrame base: {len(output_df)} rows, {len(output_df.columns)} columns")
        
        # Add Abnormality_score column
        output_df['Abnormality_score'] = abnormality_scores.astype(float)
        
        # Add top feature columns
        feature_columns = [f'top_feature_{i+1}' for i in range(self.config.max_top_features)]
        for i, col in enumerate(feature_columns):
            if i < len(top_features_df.columns):
                output_df[col] = top_features_df.iloc[:, i].values
            else:
                output_df[col] = ""  # Fill missing columns with empty strings
        
        print(f"Final output DataFrame: {len(output_df)} rows, {len(output_df.columns)} columns")
        print(f"New columns added: Abnormality_score + {len(feature_columns)} feature columns")
        
        return output_df
    
    def validate_output_format(self, df: pd.DataFrame) -> bool:
        """
        Validate that output DataFrame meets hackathon specifications.
        
        Args:
            df: Output DataFrame to validate
            
        Returns:
            True if format is valid, False otherwise
        """
        print("Validating output format...")
        
        issues = []
        
        # Check required columns exist
        required_columns = ['Abnormality_score'] + [f'top_feature_{i+1}' for i in range(7)]
        
        for col in required_columns:
            if col not in df.columns:
                issues.append(f"Missing required column: {col}")
        
        # Validate Abnormality_score column
        if 'Abnormality_score' in df.columns:
            score_col = df['Abnormality_score']
            
            # Check data type
            if not pd.api.types.is_numeric_dtype(score_col):
                issues.append("Abnormality_score must be numeric")
            else:
                # Check value range
                if score_col.min() < 0.0 or score_col.max() > 100.0:
                    issues.append(f"Abnormality_score out of range [0,100]: {score_col.min():.2f} to {score_col.max():.2f}")
                
                # Check for NaN values
                if score_col.isnull().any():
                    issues.append("Abnormality_score contains NaN values")
        
        # Validate top feature columns
        feature_columns = [f'top_feature_{i+1}' for i in range(7)]
        for col in feature_columns:
            if col in df.columns:
                feature_col = df[col]
                
                # Check data type (should be string)
                if not pd.api.types.is_string_dtype(feature_col) and not pd.api.types.is_object_dtype(feature_col):
                    issues.append(f"{col} should contain string values")
                
                # Check for NaN values (should be empty strings instead)
                if feature_col.isnull().any():
                    issues.append(f"{col} contains NaN values (should be empty strings)")
        
        # Check index (should be datetime)
        if not isinstance(df.index, pd.DatetimeIndex):
            issues.append("Index should be DatetimeIndex (timestamps)")
        
        # Report validation results
        if issues:
            print("Output format validation failed:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("Output format validation passed")
            return True
    
    def save_to_csv(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save DataFrame to CSV file with proper formatting.
        
        Args:
            df: DataFrame to save
            output_path: Path where CSV file will be saved
            
        Raises:
            PermissionError: If unable to write to output path
            ValueError: If DataFrame is invalid
        """
        if df.empty:
            raise ValueError("Cannot save empty DataFrame")
        
        # Validate output path
        validate_output_path(output_path)
        
        print(f"Saving output to: {output_path}")
        
        try:
            # Save with timestamp index preserved
            df.to_csv(output_path, index=True, float_format='%.6f')
            
            # Verify file was created and has content
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"Output saved successfully. File size: {file_size} bytes")
            else:
                raise ValueError("Output file was not created")
                
        except Exception as e:
            raise ValueError(f"Failed to save output CSV: {e}")
    
    def generate_summary_stats(self, df: pd.DataFrame) -> dict:
        """
        Generate summary statistics for the output data.
        
        Args:
            df: Output DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'time_range': {
                'start': str(df.index.min()),
                'end': str(df.index.max())
            }
        }
        
        # Abnormality score statistics
        if 'Abnormality_score' in df.columns:
            score_col = df['Abnormality_score']
            stats['abnormality_scores'] = {
                'mean': float(score_col.mean()),
                'std': float(score_col.std()),
                'min': float(score_col.min()),
                'max': float(score_col.max()),
                'median': float(score_col.median())
            }
            
            # Score distribution
            stats['score_distribution'] = {
                'normal_0_20': int((score_col <= 20).sum()),
                'slight_21_30': int(((score_col > 20) & (score_col <= 30)).sum()),
                'moderate_31_60': int(((score_col > 30) & (score_col <= 60)).sum()),
                'significant_61_90': int(((score_col > 60) & (score_col <= 90)).sum()),
                'severe_91_100': int((score_col > 90).sum())
            }
        
        # Feature attribution statistics
        feature_columns = [f'top_feature_{i+1}' for i in range(7)]
        feature_stats = {}
        
        for col in feature_columns:
            if col in df.columns:
                non_empty = (df[col] != "").sum()
                feature_stats[col] = {
                    'non_empty_count': int(non_empty),
                    'empty_count': int(len(df) - non_empty)
                }
        
        stats['feature_attribution'] = feature_stats
        
        return stats
    
    def process_output_generation(self, original_data: pd.DataFrame,
                                analysis_data: pd.DataFrame,
                                abnormality_scores: np.ndarray,
                                top_features_df: pd.DataFrame,
                                output_path: str,
                                binary_predictions: Optional[np.ndarray] = None) -> dict:
        """
        Complete output generation pipeline.
        
        Args:
            original_data: Original CSV data
            analysis_data: Processed analysis data
            abnormality_scores: Anomaly scores
            top_features_df: Top contributing features
            output_path: Path to save output CSV
            binary_predictions: Optional binary predictions
            
        Returns:
            Dictionary with summary statistics
        """
        print("Starting output generation pipeline...")
        
        # Create output DataFrame
        output_df = self.create_output_dataframe(
            original_data, analysis_data, abnormality_scores, 
            top_features_df, binary_predictions
        )
        
        # Validate format
        if not self.validate_output_format(output_df):
            raise ValueError("Output format validation failed")
        
        # Save to CSV
        self.save_to_csv(output_df, output_path)
        
        # Generate summary statistics
        summary_stats = self.generate_summary_stats(output_df)
        
        print("Output generation pipeline completed successfully")
        
        return summary_stats