"""
Test script for the anomaly detection system.
"""

import sys
import os
from anomaly_detection_system.main import main

def test_with_tep_dataset():
    """Test the system with the TEP dataset."""
    
    input_file = "TEP_Train_Test.csv"
    output_file = "anomaly_results.csv"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Please ensure TEP_Train_Test.csv is in the current directory.")
        return False
    
    try:
        print("Testing Anomaly Detection System with TEP dataset...")
        print("=" * 50)
        
        # Run the system
        stats = main(input_file, output_file)
        
        print("\nTest Results:")
        print("-" * 20)
        print(f"Success: {stats['success']}")
        print(f"Runtime: {stats['total_runtime_minutes']:.2f} minutes")
        print(f"Training Validation: {stats['training_validation_passed']}")
        
        if 'output_statistics' in stats:
            output_stats = stats['output_statistics']
            print(f"Output rows: {output_stats['total_rows']}")
            print(f"Output columns: {output_stats['total_columns']}")
            
            if 'abnormality_scores' in output_stats:
                score_stats = output_stats['abnormality_scores']
                print(f"Score range: {score_stats['min']:.2f} - {score_stats['max']:.2f}")
                print(f"Score mean: {score_stats['mean']:.2f}")
        
        print(f"\nOutput saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_with_tep_dataset()
    sys.exit(0 if success else 1)