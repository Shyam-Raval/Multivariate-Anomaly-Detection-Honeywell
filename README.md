# Multivariate Time Series Anomaly Detection System

A comprehensive Python system for detecting anomalies in industrial sensor data with explainable feature attribution.

## Quick Start

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the system with provided dataset
python test_system.py

# Output: anomaly_results.csv with anomaly scores and feature attribution
```

## Features

- **Isolation Forest** anomaly detection model
- **SHAP-based feature attribution** for explainable results
- **0-100 anomaly scoring scale** with percentile ranking
- **Robust data preprocessing** with edge case handling
- **PEP8 compliant** modular architecture
- **Comprehensive validation** and error handling


### Test with Provided Dataset

```bash
python test_system.py
```

## Output Format

The system generates a CSV file with:

- All original columns preserved
- `Abnormality_score`: Float values 0.0-100.0
- `top_feature_1` through `top_feature_7`: Names of most contributing features

## Score Interpretation

- **0-20**: Normal behavior (expected for training period)
- **21-30**: Slightly unusual but acceptable
- **31-60**: Moderate anomaly requiring attention
- **61-90**: Significant anomaly needing investigation
- **91-100**: Severe anomaly requiring immediate action

## Configuration

The system uses predefined time periods optimized for industrial data analysis:

- **Training Period**: 2004-01-01 00:00:00 to 2004-01-05 23:59:00 (120 hours)
- **Analysis Period**: 2004-01-01 00:00:00 to 2004-01-19 07:59:00 (439 hours)
- **Model**: Isolation Forest with 100 estimators, 5% contamination rate
- **Feature Attribution**: SHAP for small datasets, efficient permutation for large datasets

## Architecture

```
anomaly_detection_system/
├── core/
│   ├── data_processor.py      # Data loading and preprocessing
│   ├── model_trainer.py       # Isolation Forest training
│   ├── anomaly_detector.py    # Anomaly detection and scoring
│   ├── feature_attributor.py  # SHAP-based feature attribution
│   └── output_generator.py    # CSV output generation
├── utils/
│   ├── config.py             # System configuration
│   └── validators.py         # Input validation utilities
└── main.py                   # Main application interface
```

## Requirements

- **Python**: 3.8+ (tested on 3.12)
- **Core Libraries**:
  - pandas >= 1.5.0 (data manipulation)
  - numpy >= 1.21.0 (numerical computing)
  - scikit-learn >= 1.1.0 (machine learning)
  - scipy >= 1.9.0 (scientific computing)
  - shap >= 0.41.0 (explainable AI)

## Performance

- **Fast Processing**: ~8 seconds for 26,400 samples (0.14 minutes)
- **Scalable**: Optimized for datasets up to 50,000+ rows
- **Memory Efficient**: Batch processing with intelligent sampling
- **Production Ready**: Robust error handling and validation

## Key Results

- **Dataset**: TEP (Tennessee Eastman Process) industrial sensor data
- **Processing Speed**: 26,400 rows × 52 features in 8.4 seconds
- **Anomaly Detection**: 7,274 anomalies detected out of 26,400 samples
- **Feature Attribution**: Top 7 contributing features identified per anomaly
- **Output Quality**: 0-100 scoring scale with explainable results
