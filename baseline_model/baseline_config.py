#!/usr/bin/env python3
"""
Baseline Configuration for MSc Thesis Final Project
Simple CNN-LSTM model targeting R² ≈ 0.7
"""

import os
from datetime import datetime


class Config:
    """Simple baseline configuration for CNN-LSTM model"""

    # =====================================================================================
    # PATHS AND GENERAL SETTINGS
    # =====================================================================================

    # Data path - Updated for new project
    DATA_PATH = r'C:\Users\Fan\PycharmProjects\Msc_thesis_final\processed_data\preprocessed_milan_traffic_center_1600_7day_splits_optimized.pkl'

    # Output directory
    OUTPUT_BASE_DIR = 'baseline_outputs'

    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FREQUENCY = 50

    # Device settings
    USE_CUDA = True

    # Windows memory optimization
    NUM_WORKERS = 0  # Disable multiprocessing for Windows stability
    PIN_MEMORY = False  # Disable pin memory on Windows
    PREFETCH_FACTOR = None
    PERSISTENT_WORKERS = False

    # =====================================================================================
    # SIMPLE MODEL ARCHITECTURE
    # =====================================================================================

    # Model dimensions (very basic for proper baseline performance)
    HIDDEN_DIM = 64  # Much smaller capacity
    NUM_LAYERS = 1  # Single LSTM layer only
    PREDICTION_HORIZON = 12

    # Simple features
    USE_BATCH_NORM = False  # Remove BatchNorm for simplicity
    USE_DROPOUT = True
    DROPOUT_RATE = 0.3  # High dropout for more regularization

    # =====================================================================================
    # CENTER REGION SETTINGS
    # =====================================================================================

    # Grid information
    MILAN_GRID_SIZE = 100
    CENTER_REGION_SIZE = 40
    TOTAL_CENTER_SQUARES = 1600

    # Center region boundaries
    CENTER_START_ROW = 30
    CENTER_END_ROW = 69
    CENTER_START_COL = 30
    CENTER_END_COL = 69

    # =====================================================================================
    # DATA SPLIT INFORMATION
    # =====================================================================================

    # Split dates
    VALIDATION_START_DATE = "2013-12-09"
    VALIDATION_END_DATE = "2013-12-16"
    TEST_START_DATE = "2013-12-17"
    TEST_END_DATE = "2013-12-23"

    # Split descriptions
    SPLIT_DESCRIPTION = {
        'train': 'Nov 1 - Dec 8 (~70%)',
        'val': 'Dec 9 - Dec 16 (~15%)',
        'test': 'Dec 17 - Dec 23 (~15%)'
    }

    DATA_EXCLUSIONS = "Christmas period (Dec 24-31) completely excluded from dataset"

    # =====================================================================================
    # TRAINING SETTINGS
    # =====================================================================================

    # Basic training parameters (conservative for weak baseline)
    BATCH_SIZE = 64  # Much larger batches for noisier gradients
    NUM_EPOCHS = 30  # Very few epochs to prevent overtraining
    LEARNING_RATE = 0.005  # Higher learning rate for less precise convergence
    WEIGHT_DECAY = 5e-3  # Strong regularization

    # Early stopping (very aggressive)
    PATIENCE = 8  # Very early stopping
    MIN_DELTA = 1e-6

    # Gradient clipping
    GRADIENT_CLIP_VALUE = 1.0

    # Model saving
    SAVE_BEST_MODEL = True
    SAVE_CHECKPOINTS = True
    CHECKPOINT_FREQUENCY = 10

    # =====================================================================================
    # THESIS DATA COLLECTION SETTINGS
    # =====================================================================================

    # Comprehensive metrics tracking
    TRACK_COMPREHENSIVE_METRICS = True
    TRACK_DAY_SPECIFIC_METRICS = True
    TRACK_HOUR_SPECIFIC_METRICS = True
    TRACK_HORIZON_SPECIFIC_METRICS = True
    TRACK_SPATIAL_METRICS = True

    # Visualization settings for thesis
    VIZ_SAVE_DPI = 300
    SAVE_HIGH_QUALITY_PLOTS = True
    SAVE_DETAILED_ANALYSIS = True

    # Export formats for thesis
    EXPORT_FORMATS = ['png', 'pdf', 'svg']

    # Thesis-specific analysis
    GENERATE_THESIS_REPORT = True
    SAVE_LATEX_TABLES = True
    SAVE_RAW_DATA = True

    @classmethod
    def create_output_folder(cls):
        """Create timestamped output folder for baseline experiments"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(cls.OUTPUT_BASE_DIR, f'baseline_simple_cnn_lstm_{timestamp}')

        # Create subdirectories
        directories = [
            'models',
            'checkpoints',
            'plots',
            'metrics',
            'predictions',
            'thesis_data',
            'latex_tables',
            'raw_data'
        ]

        os.makedirs(output_dir, exist_ok=True)
        for directory in directories:
            os.makedirs(os.path.join(output_dir, directory), exist_ok=True)

        return output_dir

    @classmethod
    def save_config(cls, output_dir):
        """Save configuration to file"""
        import json

        config_dict = {}
        for key in dir(cls):
            if not key.startswith('_') and not callable(getattr(cls, key)):
                value = getattr(cls, key)
                if isinstance(value, (int, float, str, bool, list, dict)):
                    config_dict[key] = value
                else:
                    config_dict[key] = str(value)

        config_path = os.path.join(output_dir, 'baseline_config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)

        return config_path

    @classmethod
    def get_center_region_info(cls):
        """Get information about the center region"""
        return {
            'grid_size': f"{cls.MILAN_GRID_SIZE}x{cls.MILAN_GRID_SIZE}",
            'center_region_size': f"{cls.CENTER_REGION_SIZE}x{cls.CENTER_REGION_SIZE}",
            'total_squares': cls.TOTAL_CENTER_SQUARES,
            'coverage_percentage': (cls.TOTAL_CENTER_SQUARES / (cls.MILAN_GRID_SIZE ** 2)) * 100,
            'boundaries': {
                'rows': f"{cls.CENTER_START_ROW} to {cls.CENTER_END_ROW}",
                'cols': f"{cls.CENTER_START_COL} to {cls.CENTER_END_COL}"
            }
        }

    @classmethod
    def get_model_info(cls):
        """Get model architecture information"""
        return {
            'model_type': 'Simple CNN-LSTM Baseline',
            'architecture': 'CNN + Bidirectional LSTM',
            'hidden_dim': cls.HIDDEN_DIM,
            'num_layers': cls.NUM_LAYERS,
            'prediction_horizon': cls.PREDICTION_HORIZON,
            'target_performance': 'R² ≈ 0.7',
            'total_parameters_estimate': cls.estimate_parameters()
        }

    @classmethod
    def estimate_parameters(cls):
        """Estimate total model parameters"""
        # Rough estimate for documentation
        n_squares = cls.TOTAL_CENTER_SQUARES
        n_temporal = 11  # Standard temporal features
        hidden_dim = cls.HIDDEN_DIM

        # CNN parameters
        cnn_params = (n_squares * hidden_dim * 3) + (hidden_dim * hidden_dim * 3)

        # LSTM parameters (bidirectional)
        lstm_params = 4 * hidden_dim * (hidden_dim + hidden_dim + 1) * 2 * cls.NUM_LAYERS

        # Dense layers
        dense_params = (n_temporal * hidden_dim // 2) + (hidden_dim * 3 * hidden_dim) + (
                    hidden_dim * n_squares * cls.PREDICTION_HORIZON)

        total_estimate = cnn_params + lstm_params + dense_params
        return f"~{total_estimate // 1000}K parameters"


# Alternative configurations for different performance targets
class HighPerformanceConfig(Config):
    """Higher capacity configuration for better performance"""

    HIDDEN_DIM = 512
    NUM_LAYERS = 3
    BATCH_SIZE = 8  # Smaller batch for larger model
    NUM_EPOCHS = 150
    LEARNING_RATE = 0.0005


class FastConfig(Config):
    """Faster training configuration for quick experiments"""

    HIDDEN_DIM = 128
    NUM_LAYERS = 1
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.002


# Use the main Config class as default
BaselineConfig = Config