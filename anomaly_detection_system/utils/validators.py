"""
Validation utilities for input data and system parameters.
"""

import os
from pathlib import Path
from typing import Optional
import pandas as pd


def validate_csv_path(csv_path: str) -> bool:
    """
    Validate that the CSV file path exists and is readable.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file is not readable
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    if not os.access(csv_path, os.R_OK):
        raise PermissionError(f"Cannot read CSV file: {csv_path}")
    
    # Check if it's actually a file (not a directory)
    if not os.path.isfile(csv_path):
        raise ValueError(f"Path is not a file: {csv_path}")
    
    return True


def validate_timestamp_format(timestamp_str: str) -> bool:
    """
    Validate timestamp format matches expected pattern.
    
    Args:
        timestamp_str: Timestamp string to validate
        
    Returns:
        True if valid format, False otherwise
    """
    try:
        pd.to_datetime(timestamp_str, format='%m/%d/%Y %H:%M')
        return True
    except (ValueError, TypeError):
        try:
            # Try alternative format
            pd.to_datetime(timestamp_str, format='%Y-%m-%d %H:%M:%S')
            return True
        except (ValueError, TypeError):
            return False


def validate_output_path(output_path: str) -> bool:
    """
    Validate that the output path is writable.
    
    Args:
        output_path: Path where output CSV will be saved
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        PermissionError: If directory is not writable
    """
    output_dir = Path(output_path).parent
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if directory is writable
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"Cannot write to directory: {output_dir}")
    
    return True