#!/usr/bin/env python3
"""
Simple Main Script to Run Baseline Model Training
"""

import os
import sys

# Fix OpenMP conflicts before importing other libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import our modules
from baseline_config import Config
from baseline_model import train_baseline_model


def main():
    """
    Main function to run baseline model training
    """

    print("=" * 80)
    print("MSC THESIS: BASELINE MODEL TRAINING")
    print("=" * 80)
    print("Project: Traffic Prediction for Milan Center (1600 squares)")
    print("Model: Simple CNN-LSTM Baseline")
    print("Target: R² ≈ 0.7")
    print("=" * 80)

    # Check if data file exists
    data_path = Config.DATA_PATH
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found at {data_path}")
        print("Please ensure the preprocessed data file exists at the specified path.")
        return None

    print(f"✅ Data file found: {data_path}")

    # Display configuration summary
    print(f"\nConfiguration Summary:")
    print(f"  Hidden Dimension: {Config.HIDDEN_DIM}")
    print(f"  Number of Layers: {Config.NUM_LAYERS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Number of Epochs: {Config.NUM_EPOCHS}")
    print(f"  Learning Rate: {Config.LEARNING_RATE}")
    print(f"  Prediction Horizon: {Config.PREDICTION_HORIZON} steps (2 hours)")
    print(f"  Use CUDA: {Config.USE_CUDA}")

    # Display center region info
    center_info = Config.get_center_region_info()
    print(f"\nCenter Region Info:")
    print(f"  Grid Size: {center_info['grid_size']}")
    print(f"  Center Region: {center_info['center_region_size']}")
    print(f"  Total Squares: {center_info['total_squares']:,}")
    print(f"  Coverage: {center_info['coverage_percentage']:.1f}% of total grid")
    print(f"  Boundaries: {center_info['boundaries']}")

    try:
        print(f"\n🚀 Starting baseline model training...")

        # Train the baseline model
        results, output_dir = train_baseline_model(Config)

        if results is not None:
            # Training completed successfully
            print(f"\n" + "=" * 80)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 80)

            print(f"Final Results:")
            print(f"  Test R²: {results['test_r2']:.4f}")
            print(f"  Test MAE: {results['test_mae']:.4f}")
            print(f"  Test RMSE: {results['test_rmse']:.4f}")
            print(f"  Overprediction Rate: {results['test_overprediction_rate']:.1f}%")
            print(f"  Total Parameters: {results['total_parameters']:,}")
            print(f"  Epochs Trained: {results['epochs_trained']}")

            print(f"\nOutput Directory: {output_dir}")
            print(f"Model saved at: {results['model_path']}")

            # Next steps
            print(f"\n📊 Next Steps:")
            print(f"1. Run baseline_visualization.py to generate comprehensive plots")
            print(f"2. Use the saved model for further experiments")
            print(f"3. Compare with advanced models")

            # Performance guidance
            if results['test_r2'] >= 0.7:
                print(f"\n🎯 Target R² achieved! This baseline is ready for comparison.")
            elif results['test_r2'] >= 0.5:
                print(f"\n👍 Strong baseline performance. Consider minor tuning if needed.")
            else:
                print(f"\n💡 Suggestions for improvement:")
                print(f"   - Increase hidden_dim or num_layers in config")
                print(f"   - Train for more epochs")
                print(f"   - Adjust learning rate")

            return results, output_dir

        else:
            print(f"\n❌ Training failed - no results generated")
            return None, None

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    import torch

    # Set number of threads for stability
    torch.set_num_threads(1)

    # Run main function
    results, output_dir = main()

    if results is not None:
        print(f"\n✅ Baseline training completed successfully!")
        print(f"Check {output_dir} for all outputs.")
    else:
        print(f"\n❌ Baseline training failed!")
        sys.exit(1)
