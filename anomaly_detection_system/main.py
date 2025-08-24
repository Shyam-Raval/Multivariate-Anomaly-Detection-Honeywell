"""
Main application module for the Multivariate Time Series Anomaly Detection System.
"""

import argparse
import sys
import time
import traceback
from typing import Optional, Dict, Any
import pandas as pd

from .utils.config import SystemConfig
from .core.data_processor import DataProcessor
from .core.model_trainer import ModelTrainer
from .core.anomaly_detector import AnomalyDetector
from .core.feature_attributor import FeatureAttributor
from .core.output_generator import OutputGenerator


class AnomalyDetectionSystem:
    """
    Main system class that orchestrates the complete anomaly detection pipeline.
    
    This class coordinates all components to:
    - Load and preprocess data
    - Train anomaly detection model
    - Detect anomalies and calculate scores
    - Perform feature attribution
    - Generate formatted output
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """
        Initialize the anomaly detection system.
        
        Args:
            config: System configuration. If None, uses default configuration.
        """
        self.config = config or SystemConfig()
        
        # Initialize components
        self.data_processor = DataProcessor(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.feature_attributor = FeatureAttributor(self.config)
        self.output_generator = OutputGenerator(self.config)
        
        # Runtime tracking
        self.start_time: Optional[float] = None
        self.processing_stats: Dict[str, Any] = {}
    
    def run(self, input_csv_path: str, output_csv_path: str) -> Dict[str, Any]:
        """
        Execute the complete anomaly detection pipeline.
        
        Args:
            input_csv_path: Path to input CSV file
            output_csv_path: Path where output CSV will be saved
            
        Returns:
            Dictionary with processing statistics and results
            
        Raises:
            ValueError: If processing fails at any stage
            RuntimeError: If runtime exceeds maximum allowed time
        """
        self.start_time = time.time()
        
        try:
            print("="*60)
            print("MULTIVARIATE TIME SERIES ANOMALY DETECTION SYSTEM")
            print("="*60)
            print(f"Input: {input_csv_path}")
            print(f"Output: {output_csv_path}")
            print(f"Max Runtime: {self.config.max_runtime_minutes} minutes")
            print()
            
            # Stage 1: Data Loading and Preprocessing
            print("STAGE 1: DATA LOADING AND PREPROCESSING")
            print("-" * 40)
            stage_start = time.time()
            
            original_data = self.data_processor.load_and_validate(input_csv_path)
            processed_data = self.data_processor.preprocess_data(original_data)
            training_data, analysis_data = self.data_processor.split_training_analysis(processed_data)
            
            stage_time = time.time() - stage_start
            print(f"Stage 1 completed in {stage_time:.2f} seconds")
            print()
            
            # Check runtime
            self._check_runtime("after data preprocessing")
            
            # Stage 2: Model Training
            print("STAGE 2: MODEL TRAINING")
            print("-" * 40)
            stage_start = time.time()
            
            model = self.model_trainer.train_model(training_data)
            training_valid = self.model_trainer.validate_training_scores(model, training_data)
            
            stage_time = time.time() - stage_start
            print(f"Stage 2 completed in {stage_time:.2f} seconds")
            print()
            
            # Check runtime
            self._check_runtime("after model training")
            
            # Stage 3: Anomaly Detection and Scoring
            print("STAGE 3: ANOMALY DETECTION AND SCORING")
            print("-" * 40)
            stage_start = time.time()
            
            binary_predictions, abnormality_scores = self.anomaly_detector.process_analysis_data(
                model, analysis_data
            )
            
            stage_time = time.time() - stage_start
            print(f"Stage 3 completed in {stage_time:.2f} seconds")
            print()
            
            # Check runtime
            self._check_runtime("after anomaly detection")
            
            # Stage 4: Feature Attribution
            print("STAGE 4: FEATURE ATTRIBUTION")
            print("-" * 40)
            stage_start = time.time()
            
            top_features_df = self.feature_attributor.process_feature_attribution(
                model, analysis_data, training_data
            )
            
            stage_time = time.time() - stage_start
            print(f"Stage 4 completed in {stage_time:.2f} seconds")
            print()
            
            # Check runtime
            self._check_runtime("after feature attribution")
            
            # Stage 5: Output Generation
            print("STAGE 5: OUTPUT GENERATION")
            print("-" * 40)
            stage_start = time.time()
            
            summary_stats = self.output_generator.process_output_generation(
                original_data, analysis_data, abnormality_scores, 
                top_features_df, output_csv_path, binary_predictions
            )
            
            stage_time = time.time() - stage_start
            print(f"Stage 5 completed in {stage_time:.2f} seconds")
            print()
            
            # Final results
            total_time = time.time() - self.start_time
            print("="*60)
            print("PROCESSING COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"Total Runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            print(f"Training Validation: {'PASSED' if training_valid else 'FAILED (with warnings)'}\")\n")
            
            # Compile processing statistics
            self.processing_stats = {
                'success': True,
                'total_runtime_seconds': total_time,
                'total_runtime_minutes': total_time / 60,
                'training_validation_passed': training_valid,
                'model_parameters': self.model_trainer.get_model_parameters(),
                'data_statistics': {
                    'original_rows': len(original_data),
                    'original_columns': len(original_data.columns),
                    'numerical_features': len(self.data_processor.numerical_columns or []),
                    'training_samples': len(training_data),
                    'analysis_samples': len(analysis_data)
                },
                'output_statistics': summary_stats
            }
            
            return self.processing_stats
            
        except Exception as e:
            total_time = time.time() - (self.start_time or time.time())
            error_stats = {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'total_runtime_seconds': total_time,
                'traceback': traceback.format_exc()
            }
            
            print("="*60)
            print("PROCESSING FAILED")
            print("="*60)
            print(f"Error: {e}")
            print(f"Runtime before failure: {total_time:.2f} seconds")
            print()
            
            raise RuntimeError(f"Anomaly detection pipeline failed: {e}") from e
    
    def _check_runtime(self, stage: str) -> None:
        """
        Check if runtime is within acceptable limits.
        
        Args:
            stage: Description of current processing stage
            
        Raises:
            RuntimeError: If runtime exceeds maximum allowed time
        """
        if self.start_time is None:
            return
        
        elapsed_minutes = (time.time() - self.start_time) / 60
        max_minutes = self.config.max_runtime_minutes
        
        if elapsed_minutes > max_minutes:
            raise RuntimeError(
                f"Runtime limit exceeded {stage}: {elapsed_minutes:.2f} minutes > {max_minutes} minutes"
            )
        
        # Warning at 80% of max time
        if elapsed_minutes > max_minutes * 0.8:
            remaining = max_minutes - elapsed_minutes
            print(f"WARNING: Approaching runtime limit. {remaining:.1f} minutes remaining.")


def main(input_csv_path: str, output_csv_path: str, 
         config: Optional[SystemConfig] = None) -> Dict[str, Any]:
    """
    Main function to run the anomaly detection system.
    
    Args:
        input_csv_path: Path to input CSV file
        output_csv_path: Path where output CSV will be saved
        config: Optional system configuration
        
    Returns:
        Dictionary with processing statistics
    """
    system = AnomalyDetectionSystem(config)
    return system.run(input_csv_path, output_csv_path)


def cli_main() -> None:
    """
    Command-line interface main function.
    """
    parser = argparse.ArgumentParser(
        description="Multivariate Time Series Anomaly Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m anomaly_detection_system input.csv output.csv
  python -m anomaly_detection_system TEP_Train_Test.csv results.csv
        """
    )
    
    parser.add_argument(
        'input_csv',
        help='Path to input CSV file with time series data'
    )
    
    parser.add_argument(
        'output_csv',
        help='Path where output CSV file will be saved'
    )
    
    parser.add_argument(
        '--max-runtime',
        type=int,
        default=15,
        help='Maximum runtime in minutes (default: 15)'
    )
    
    parser.add_argument(
        '--contamination',
        type=float,
        default=0.05,
        help='Expected proportion of anomalies (default: 0.05)'
    )
    
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='Number of trees in Isolation Forest (default: 100)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    try:
        # Create custom configuration if needed
        config = SystemConfig()
        config.max_runtime_minutes = args.max_runtime
        config.model_params['contamination'] = args.contamination
        config.model_params['n_estimators'] = args.n_estimators
        
        # Run the system
        stats = main(args.input_csv, args.output_csv, config)
        
        if args.verbose:
            print("\nProcessing Statistics:")
            print("-" * 20)
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"{key}:")
                    for subkey, subvalue in value.items():
                        print(f"  {subkey}: {subvalue}")
                else:
                    print(f"{key}: {value}")
        
        print(f"\nSuccess! Output saved to: {args.output_csv}")
        sys.exit(0)
        
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    cli_main()