"""
Data processing module for loading, validating, and preprocessing time series data.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Tuple, Optional, List
from pathlib import Path

from ..utils.config import SystemConfig
from ..utils.validators import validate_csv_path, validate_timestamp_format


class DataProcessor:
    """
    Handles all data loading, validation, and preprocessing operations.
    
    This class is responsible for:
    - Loading CSV files with proper error handling
    - Validating timestamp formats and regularity
    - Preprocessing data (missing values, filtering, etc.)
    - Splitting data into training and analysis periods
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize DataProcessor with system configuration.
        
        Args:
            config: System configuration containing processing parameters
        """
        self.config = config
        self.original_columns: Optional[List[str]] = None
        self.numerical_columns: Optional[List[str]] = None
    
    def load_and_validate(self, csv_path: str) -> pd.DataFrame:
        """
        Load CSV file and perform initial validation.
        
        Args:
            csv_path: Path to the CSV file to load
            
        Returns:
            Loaded and initially validated DataFrame
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV format is invalid or required columns missing
        """
        # Validate file path
        validate_csv_path(csv_path)
        
        try:
            # Load CSV file
            df = pd.read_csv(csv_path)
            print(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Store original column names
            self.original_columns = df.columns.tolist()
            
            # Check if Time column exists
            if 'Time' not in df.columns:
                raise ValueError("CSV file must contain a 'Time' column")
            
            # Validate and convert Time column
            df = self._process_timestamps(df)
            
            # Validate timestamp regularity
            self.validate_timestamps(df)
            
            return df
            
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV file is empty: {csv_path}")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing CSV file: {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error loading CSV: {e}")
    
    def _process_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and validate timestamp column.
        
        Args:
            df: DataFrame with Time column
            
        Returns:
            DataFrame with processed timestamp index
        """
        try:
            # Try primary format first
            df['Time'] = pd.to_datetime(df['Time'], format='%m/%d/%Y %H:%M')
        except ValueError:
            try:
                # Try alternative format
                df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S')
            except ValueError:
                # Let pandas infer the format
                df['Time'] = pd.to_datetime(df['Time'])
                warnings.warn("Timestamp format not recognized, using pandas inference")
        
        # Set Time as index
        df.set_index('Time', inplace=True)
        
        return df
    
    def validate_timestamps(self, df: pd.DataFrame) -> bool:
        """
        Validate timestamp regularity and intervals.
        
        Args:
            df: DataFrame with timestamp index
            
        Returns:
            True if timestamps are valid, False otherwise
        """
        # Calculate time differences
        time_diff = df.index.to_series().diff().dropna()
        
        # Check for regular 1-minute intervals
        regular_intervals = time_diff == pd.Timedelta(minutes=1)
        
        if not regular_intervals.all():
            irregular_count = (~regular_intervals).sum()
            warnings.warn(
                f"Warning: {irregular_count} irregular time intervals detected. "
                "Expected 1-minute intervals."
            )
            return False
        else:
            print("Timestamps are valid with regular 1-minute intervals.")
            return True
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data by handling missing values and filtering columns.
        
        Args:
            df: Raw DataFrame to preprocess
            
        Returns:
            Preprocessed DataFrame ready for analysis
        """
        print("Starting data preprocessing...")
        
        # Handle missing values with forward fill
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values, applying forward fill")
            df = df.ffill()
            
            # If still missing values after forward fill, use backward fill
            remaining_missing = df.isnull().sum().sum()
            if remaining_missing > 0:
                print(f"Applying backward fill for {remaining_missing} remaining missing values")
                df = df.bfill()
        
        # Filter to numerical columns only
        numerical_df = df.select_dtypes(include=[np.number])
        self.numerical_columns = numerical_df.columns.tolist()
        
        print(f"Filtered to {len(self.numerical_columns)} numerical columns")
        
        # Check for constant features (zero variance)
        constant_features = []
        for col in numerical_df.columns:
            if numerical_df[col].var() == 0:
                constant_features.append(col)
        
        if constant_features:
            print(f"Warning: Found {len(constant_features)} constant features: {constant_features}")
            # Remove constant features
            numerical_df = numerical_df.drop(columns=constant_features)
            self.numerical_columns = [col for col in self.numerical_columns if col not in constant_features]
        
        # Check minimum feature requirement
        if len(self.numerical_columns) < 1:
            raise ValueError("No valid numerical features found after preprocessing")
        
        if len(self.numerical_columns) < 7:
            warnings.warn(f"Only {len(self.numerical_columns)} features available (less than 7)")
        
        print(f"Preprocessing complete. Final dataset: {len(numerical_df)} rows, {len(self.numerical_columns)} features")
        
        return numerical_df
    
    def split_training_analysis(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and analysis periods based on configuration.
        
        Args:
            df: Preprocessed DataFrame to split
            
        Returns:
            Tuple of (training_data, analysis_data)
        """
        training_start, training_end = self.config.training_period
        analysis_start, analysis_end = self.config.analysis_period
        
        # Extract training period
        training_data = df[training_start:training_end]
        
        # Extract analysis period  
        analysis_data = df[analysis_start:analysis_end]
        
        # Validate training data sufficiency
        training_hours = len(training_data) / 60  # Assuming 1-minute intervals
        if training_hours < 72:
            warnings.warn(
                f"Training period contains only {training_hours:.1f} hours of data. "
                "Minimum 72 hours recommended for reliable model training."
            )
        
        print(f"Training period: {len(training_data)} rows ({training_hours:.1f} hours)")
        print(f"Analysis period: {len(analysis_data)} rows ({len(analysis_data)/60:.1f} hours)")
        
        if len(training_data) == 0:
            raise ValueError("Training period contains no data")
        
        if len(analysis_data) == 0:
            raise ValueError("Analysis period contains no data")
        
        return training_data, analysis_data