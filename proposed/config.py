#!/usr/bin/env python3
"""
Complex Architecture Config for Advanced Traffic Prediction
Significantly increased model complexity with attention mechanisms
"""

import os
from datetime import datetime


class ComplexConfig:
    """Complex architecture configuration for advanced traffic prediction"""

    # =====================================================================================
    # PATHS AND GENERAL SETTINGS
    # =====================================================================================

    # Data path
    DATA_PATH = r'C:\Users\Fan\PycharmProjects\Msc_Thesis\processed_data\preprocessed_milan_traffic_center_1600_7day_splits_optimized.pkl'
    # Output directory
    OUTPUT_BASE_DIR = 'outputs'

    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FREQUENCY = 50

    # Device settings
    USE_CUDA = True

    # Memory settings - adjusted for larger model
    NUM_WORKERS = 0
    PIN_MEMORY = False
    PREFETCH_FACTOR = None
    PERSISTENT_WORKERS = False

    # =====================================================================================
    # COMPLEX MODEL ARCHITECTURE
    # =====================================================================================

    # Significantly increased model capacity
    HIDDEN_DIM = 512  # Increased from 256/512
    NUM_LAYERS = 4  # Increased from 2/3
    PREDICTION_HORIZON = 12

    # Multi-scale CNN settings
    USE_MULTI_SCALE = True
    KERNEL_SIZES = [1, 3, 5, 7, 11, 15, 21]  # Expanded kernel sizes
    USE_RESIDUAL_CNN = True  # Enable residual connections in CNN

    # Advanced attention mechanisms
    USE_ATTENTION = True
    USE_SPATIAL_ATTENTION = True
    USE_TEMPORAL_ATTENTION = True
    ATTENTION_HEADS = 8  # Multi-head attention
    ATTENTION_DROPOUT = 0.1

    # Enhanced LSTM features
    USE_BIDIRECTIONAL_LSTM = True
    USE_LSTM_ATTENTION = True
    USE_HIGHWAY_CONNECTIONS = True
    LSTM_DROPOUT = 0.2

    # Advanced embeddings
    USE_DAY_EMBEDDING = True
    USE_HOUR_EMBEDDING = True
    DAY_EMBEDDING_DIM = 384  # HIDDEN_DIM // 2
    HOUR_EMBEDDING_DIM = 192  # HIDDEN_DIM // 4

    # Complex fusion network
    USE_DEEP_FUSION = True
    FUSION_LAYERS = 3
    FUSION_DROPOUT = 0.2

    # Peak enhancement complexity
    USE_MULTI_SCALE_PEAK_DETECTION = True
    PEAK_DETECTOR_SCALES = 3
    PEAK_TYPES = 4  # Different peak categories
    USE_PEAK_CLASSIFICATION = True

    # Residual and memory mechanisms
    USE_LEARNABLE_RESIDUALS = True
    USE_MEMORY_WEIGHTS = True
    USE_DAY_ADJUSTMENTS = True

    # =====================================================================================
    # COMPLEX REGION SETTINGS
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

    # Enhanced spatial modeling
    USE_SPATIAL_CONVOLUTION = True
    SPATIAL_CONV_CHANNELS = [32, 64, 128, 256]  # Increased complexity
    SPATIAL_POOL_SIZE = 2
    USE_SPATIAL_RESIDUALS = True

    # =====================================================================================
    # TRAINING SETTINGS FOR COMPLEX MODEL
    # =====================================================================================

    # Training parameters - adjusted for larger model
    BATCH_SIZE = 6  # Reduced due to larger model
    NUM_EPOCHS = 150  # Increased for complex model
    LEARNING_RATE = 0.0001  # Lower LR for stability
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.2

    # Advanced loss function
    USE_COMPLEX_LOSS = True
    MSE_WEIGHT = 0.4
    MAE_WEIGHT = 0.3
    HUBER_WEIGHT = 0.2
    FOCAL_WEIGHT = 0.1

    # Enhanced peak loss settings
    PEAK_WEIGHT = 8.0
    VERY_HIGH_PEAK_WEIGHT = 15.0
    PEAK_UNDERESTIMATION_PENALTY = 5.0
    PEAK_THRESHOLD_PERCENTILE = 80
    VERY_HIGH_PEAK_PERCENTILE = 95

    # Day-aware loss enhancements
    USE_DAY_AWARE_LOSS = True
    OVERPRED_PENALTY_WEEKDAY = 2.0
    OVERPRED_PENALTY_WEEKEND = 2.5

    # Advanced scheduler
    USE_SCHEDULER = True
    SCHEDULER_TYPE = 'cosine_with_warmup'
    WARMUP_EPOCHS = 20  # Longer warmup for complex model
    COSINE_RESTART_PERIOD = 50

    # Gradient handling
    GRADIENT_CLIP_VALUE = 1.0
    USE_GRADIENT_ACCUMULATION = True
    ACCUMULATE_GRAD_BATCHES = 2

    # Early stopping
    PATIENCE = 40  # Increased for complex model
    MIN_DELTA = 1e-7

    # =====================================================================================
    # DAY-AWARE SETTINGS
    # =====================================================================================

    # Enhanced day-specific features
    USE_DAY_SPECIFIC_RESIDUAL = True
    USE_DAY_AWARE_CALIBRATION = True
    USE_PEAK_SPECIFIC_CALIBRATION = True

    # Weekend settings
    WEEKEND_TRAFFIC_REDUCTION_FACTOR = 0.7

    # Day traffic multipliers
    DAY_TRAFFIC_MULTIPLIERS = {
        0: 1.0,  # Monday
        1: 1.05,  # Tuesday
        2: 1.05,  # Wednesday
        3: 1.05,  # Thursday
        4: 1.1,  # Friday
        5: 0.7,  # Saturday
        6: 0.6,  # Sunday
    }

    # =====================================================================================
    # VALIDATION CONFIGURATION
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
    # ADVANCED AUGMENTATION
    # =====================================================================================

    # Sophisticated augmentation
    USE_ADVANCED_AUGMENTATION = True
    AUGMENTATION_PROBABILITY = 0.25
    MULTI_SCALE_NOISE = True
    NOISE_LEVELS = [0.02, 0.05, 0.08]
    NOISE_WEIGHTS = [0.6, 0.3, 0.1]

    # Peak-aware augmentation
    USE_PEAK_AUGMENTATION = True
    PEAK_AUGMENTATION_PROBABILITY = 0.15
    PEAK_BOOST_RANGE = 0.1

    # Temporal augmentation
    USE_TEMPORAL_JITTER = True
    TEMPORAL_JITTER_RANGE = 0.05

    # Spatial augmentation
    USE_SPATIAL_AUGMENTATION = True
    SPATIAL_NOISE_CORRELATION = 0.3
    SPATIAL_AUGMENTATION_RADIUS = 2

    # Day-specific augmentation
    USE_DAY_SPECIFIC_AUGMENTATION = True
    WEEKEND_AUGMENTATION_BOOST = 1.2

    # =====================================================================================
    # COMPLEX CALIBRATION
    # =====================================================================================

    # Multi-dimensional calibration
    USE_COMPLEX_CALIBRATION = True
    USE_DAY_CALIBRATION = True
    USE_HOUR_CALIBRATION = True
    USE_PEAK_LEVEL_CALIBRATION = True
    USE_SPATIAL_CALIBRATION = True

    # Calibration constraints
    DAY_CALIBRATION_RANGE = (0.7, 1.5)
    HOUR_CALIBRATION_RANGE = (0.8, 1.3)
    PEAK_CALIBRATION_RANGE = (0.7, 1.5)
    SPATIAL_CALIBRATION_RANGE = (0.9, 1.1)

    # Calibration strength
    CALIBRATION_STRENGTH = 0.85
    DAY_CALIBRATION_STRENGTH = 0.9
    PEAK_CALIBRATION_STRENGTH = 1.15

    # =====================================================================================
    # MEMORY OPTIMIZATION
    # =====================================================================================

    # Memory management for complex model
    USE_GRADIENT_CHECKPOINTING = True
    USE_MIXED_PRECISION = False  # Disable on Windows
    USE_EFFICIENT_ATTENTION = True
    ATTENTION_CHUNK_SIZE = 100

    # Model optimization
    USE_MODEL_PARALLEL = False
    USE_DATA_PARALLEL = True

    # =====================================================================================
    # MONITORING AND EVALUATION
    # =====================================================================================

    # Enhanced metrics tracking
    TRACK_DAY_SPECIFIC_METRICS = True
    TRACK_HOUR_SPECIFIC_METRICS = True
    TRACK_PEAK_SPECIFIC_METRICS = True
    TRACK_SPATIAL_METRICS = True

    # Advanced tracking
    TRACK_ATTENTION_WEIGHTS = True
    TRACK_PEAK_DETECTION_ACCURACY = True
    TRACK_MULTI_SCALE_CONTRIBUTIONS = True
    TRACK_LAYER_ACTIVATIONS = False  # Disable to save memory

    # Visualization settings
    VIZ_SAVE_DPI = 300
    SAVE_ATTENTION_HEATMAPS = True
    SAVE_PEAK_ANALYSIS = True
    SAVE_MULTI_SCALE_ANALYSIS = True
    SAVE_SPATIAL_HEATMAPS = True

    # Model checkpointing
    SAVE_BEST_MODEL = True
    SAVE_CHECKPOINTS = True
    CHECKPOINT_FREQUENCY = 25
    SAVE_OPTIMIZER_STATE = True

    # =====================================================================================
    # INFERENCE SETTINGS
    # =====================================================================================

    # Advanced inference
    USE_ENSEMBLE_PREDICTION = True
    ENSEMBLE_WEIGHTS = [0.4, 0.3, 0.2, 0.1]  # Different model components

    # Peak boosting during inference
    USE_INFERENCE_PEAK_BOOSTING = True
    INFERENCE_PEAK_BOOST_FACTORS = [1.5, 2.0, 2.5, 3.0]

    # Day-specific ensemble
    USE_DAY_SPECIFIC_ENSEMBLE = True
    WEEKEND_MODEL_WEIGHT = 1.2

    # Post-processing
    USE_SMOOTHING = True
    SMOOTHING_WINDOW = 3
    SMOOTHING_WEIGHT = 0.1

    # Spatial smoothing
    USE_SPATIAL_SMOOTHING = True
    SPATIAL_SMOOTHING_KERNEL_SIZE = 3
    SPATIAL_SMOOTHING_WEIGHT = 0.05

    @classmethod
    def create_output_folder(cls):
        """Create timestamped output folder for complex model experiments"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(cls.OUTPUT_BASE_DIR, f'complex_center_1600_{timestamp}')

        # Create subdirectories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'metrics'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'attention_analysis'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'peak_analysis'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'calibration'), exist_ok=True)

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

        config_path = os.path.join(output_dir, 'complex_config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)

        return config_path

    @classmethod
    def get_model_complexity_info(cls):
        """Get information about model complexity"""
        return {
            'architecture_type': 'Complex Multi-Scale Attention LSTM',
            'hidden_dim': cls.HIDDEN_DIM,
            'num_layers': cls.NUM_LAYERS,
            'attention_heads': cls.ATTENTION_HEADS,
            'kernel_sizes': cls.KERNEL_SIZES,
            'peak_types': cls.PEAK_TYPES,
            'fusion_layers': cls.FUSION_LAYERS,
            'estimated_parameters': '~5-10M parameters',
            'complexity_level': 'High',
            'memory_requirements': 'High (8GB+ GPU recommended)'
        }

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


# Compatibility aliases and specialized configs
class Config(ComplexConfig):
    """Main config alias pointing to complex configuration"""
    pass


class WindowsConfig(ComplexConfig):
    """Windows-optimized complex configuration"""

    # Reduced settings for Windows compatibility
    BATCH_SIZE = 4  # Further reduced for Windows
    HIDDEN_DIM = 512  # Reduced from 768
    NUM_LAYERS = 3  # Reduced from 4
    ATTENTION_HEADS = 4  # Reduced from 8

    # Disable some memory-intensive features
    USE_GRADIENT_CHECKPOINTING = False
    USE_MIXED_PRECISION = False
    SAVE_ATTENTION_HEATMAPS = False
    TRACK_ATTENTION_WEIGHTS = False
    USE_SPATIAL_AUGMENTATION = False
    USE_SPATIAL_CALIBRATION = False

    # Conservative settings
    NUM_EPOCHS = 100
    PATIENCE = 30
    WARMUP_EPOCHS = 15


class ProductionConfig(ComplexConfig):
    """Production-ready complex configuration"""

    # Optimized for deployment
    USE_ENSEMBLE_PREDICTION = True
    USE_COMPLEX_CALIBRATION = True
    USE_INFERENCE_PEAK_BOOSTING = True

    # Disable training-only features
    USE_ADVANCED_AUGMENTATION = False
    SAVE_ATTENTION_HEATMAPS = False
    TRACK_LAYER_ACTIVATIONS = False
    TRACK_ATTENTION_WEIGHTS = False


class ExperimentalConfig(ComplexConfig):
    """Experimental configuration with maximum complexity"""

    # Push complexity to the limit
    HIDDEN_DIM = 1024
    NUM_LAYERS = 5
    ATTENTION_HEADS = 12
    FUSION_LAYERS = 4

    # Extended training
    NUM_EPOCHS = 200
    PATIENCE = 60
    WARMUP_EPOCHS = 30

    # All features enabled
    SAVE_ATTENTION_HEATMAPS = True
    TRACK_ATTENTION_WEIGHTS = True
    TRACK_LAYER_ACTIVATIONS = True
    USE_MODEL_PARALLEL = True


class LightweightConfig(ComplexConfig):
    """Lighter version for resource-constrained environments"""

    # Minimal complexity
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    ATTENTION_HEADS = 2
    FUSION_LAYERS = 2

    # Simplified features
    KERNEL_SIZES = [3, 5, 7]
    PEAK_TYPES = 2
    PEAK_DETECTOR_SCALES = 2

    # Faster training
    NUM_EPOCHS = 80
    BATCH_SIZE = 8
    PATIENCE = 20


# Legacy compatibility
class LSTMConfig(ComplexConfig):
    """LSTM configuration alias"""
    pass


class AttentionConfig(ComplexConfig):
    """Attention configuration alias"""
    pass


class TrainingConfig(ComplexConfig):
    """Training configuration alias"""
    pass


class CenterRegionConfig(ComplexConfig):
    """Center region configuration alias"""
    pass


class PeakEnhancedCenterConfig(ComplexConfig):
    """Peak-enhanced center configuration alias"""
    pass


# Default export
DEFAULT_CONFIG = WindowsConfig  # Use WindowsConfig as default for stability