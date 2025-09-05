#!/usr/bin/env python3
"""
Comprehensive Visualization Script for Baseline Traffic Prediction Model
Generates all essential figures for thesis writing including:
- True vs Predicted timeseries for validation and test sets
- Performance metrics analysis
- Day-specific performance breakdown
- Error distribution analysis
- Correlation plots

UPDATED: Larger font sizes and consistent formatting
"""

import os
# Fix OpenMP conflicts before importing other libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import gc
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')

# ========== UNIFIED CONFIGURATION SETTINGS ==========
FONT_CONFIG = {
    'base': 20,           # Increased from 16
    'title': 26,          # Increased from 20
    'subtitle': 22,       # Increased from 18
    'label': 20,          # Increased from 16
    'tick': 17,           # Increased from 14
    'legend': 17,         # Increased from 14
    'annotation': 16,     # Increased from 13
    'day_label': 19,      # Increased from 15
    'table': 15           # Increased from 12
}

# Consistent color scheme
COLOR_CONFIG = {
    'actual': '#1f77b4',      # Blue
    'predicted': '#d62728',    # Red
    'validation': '#2E86AB',
    'test': '#A23B72',
    'scatter': '#2E86AB',
    'error_scatter': '#A23B72',
    'grid_alpha': 0.3,
    'boundary': 'gray',
    'reference': 'red',
    'highlight': 'wheat',
    'header': '#40466e'
}

# Figure sizes
FIGURE_SIZES = {
    'timeseries_full': (24, 14),
    'timeseries_detail': (20, 10),
    'scatter': (16, 6),
    'day_analysis': (16, 12),
    'error_analysis': (16, 10),
    'horizon_analysis': (18, 5),
    'summary_table': (12, 6)
}

# Set style for publication-quality plots with larger fonts
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Apply font configuration globally
plt.rcParams.update({
    'font.size': FONT_CONFIG['base'],
    'axes.titlesize': FONT_CONFIG['title'],
    'axes.labelsize': FONT_CONFIG['label'],
    'xtick.labelsize': FONT_CONFIG['tick'],
    'ytick.labelsize': FONT_CONFIG['tick'],
    'legend.fontsize': FONT_CONFIG['legend'],
    'figure.titlesize': FONT_CONFIG['title'],
    'font.weight': 'normal',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'normal',
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': COLOR_CONFIG['grid_alpha'],
    'axes.axisbelow': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'gray'
})


class SimpleBaselineLSTM(nn.Module):
    """
    Simple but effective CNN-LSTM baseline model
    """

    def __init__(self, n_squares=1600, n_temporal_features=11, hidden_dim=256, num_layers=2, prediction_horizon=12):
        super(SimpleBaselineLSTM, self).__init__()

        self.n_squares = n_squares
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon

        print(f"Initializing Simple Baseline LSTM:")
        print(f"  n_squares: {n_squares}")
        print(f"  n_temporal_features: {n_temporal_features}")
        print(f"  hidden_dim: {hidden_dim}")
        print(f"  num_layers: {num_layers}")
        print(f"  prediction_horizon: {prediction_horizon}")

        # Very simple CNN for spatial feature extraction (minimal)
        self.traffic_cnn = nn.Sequential(
            nn.Conv1d(n_squares, hidden_dim, kernel_size=5, padding=2),  # Larger kernel, less precise
            nn.ReLU(),
            nn.Dropout(0.3)  # High dropout
        )

        # Simple unidirectional LSTM (minimal capacity)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.4 if num_layers > 1 else 0,  # Very high dropout
            bidirectional=False  # Unidirectional only
        )

        # Very simple temporal feature encoder
        self.temporal_encoder = nn.Sequential(
            nn.Linear(n_temporal_features, hidden_dim // 4),  # Much smaller
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Simple fusion layer
        fusion_input_dim = hidden_dim + hidden_dim // 4  # Unidirectional LSTM + small temporal
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim // 2),  # Smaller fusion
            nn.ReLU(),
            nn.Dropout(0.4)  # Very high dropout
        )

        # Very simple output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim // 2, n_squares * prediction_horizon),  # Direct output, no intermediate layers
            nn.ReLU()  # Just ensure non-negative
        )

        print("Simple Baseline LSTM initialized successfully!")

    def forward(self, x_traffic, x_temporal):
        batch_size, seq_len, _ = x_traffic.shape

        # Extract day of week for compatibility
        day_of_week = torch.argmax(x_temporal[:, -1, :7], dim=1)

        # CNN processing
        x_traffic_transposed = x_traffic.transpose(1, 2)  # (batch, features, time)
        traffic_features = self.traffic_cnn(x_traffic_transposed)
        traffic_features = traffic_features.transpose(1, 2)  # (batch, time, features)

        # LSTM processing
        lstm_out, _ = self.lstm(traffic_features)
        final_hidden = lstm_out[:, -1, :]  # Last timestep (unidirectional)

        # Temporal features
        temporal_features = self.temporal_encoder(x_temporal[:, -1, :])

        # Fusion
        combined = torch.cat([final_hidden, temporal_features], dim=1)
        fused = self.fusion(combined)

        # Generate predictions
        output = self.output_layer(fused)
        output = F.relu(output)  # Ensure non-negative

        # Reshape to (batch, horizon, squares)
        output = output.view(batch_size, self.prediction_horizon, self.n_squares)

        return output, day_of_week


class BaselineModelEvaluator:
    """Comprehensive evaluator for baseline traffic prediction models"""

    def __init__(self, data_path, model_path, output_dir=None):
        self.data_path = data_path
        self.model_path = model_path
        self.output_dir = output_dir or "baseline_evaluation_results"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

        # Load data and model
        self.data = self._load_data()
        self.model = self._load_model()

        # Generate predictions
        self.results = self._generate_predictions()

    def _load_data(self):
        """Load preprocessed data"""
        print(f"Loading data from: {self.data_path}")
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)

        print("Data loaded successfully!")
        print(f"  Train samples: {len(data['X_traffic_train'])}")
        print(f"  Validation samples: {len(data['X_traffic_val'])}")
        print(f"  Test samples: {len(data['X_traffic_test'])}")
        print(f"  Squares: {data['X_traffic_train'].shape[2]}")
        print(f"  Prediction horizon: {data['y_train'].shape[1]}")

        return data

    def _load_model(self):
        """Load trained baseline model"""
        print(f"Loading baseline model from: {self.model_path}")

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Get model parameters from data
        n_squares = self.data['X_traffic_train'].shape[2]
        n_temporal_features = self.data['X_temporal_train'].shape[2]

        # Get architecture parameters from config if available
        config_dict = checkpoint.get('config', {})
        hidden_dim = config_dict.get('HIDDEN_DIM', 64)
        num_layers = config_dict.get('NUM_LAYERS', 1)
        prediction_horizon = config_dict.get('PREDICTION_HORIZON', 12)

        print(f"Model configuration from checkpoint:")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Num layers: {num_layers}")
        print(f"  Prediction horizon: {prediction_horizon}")

        # Create model
        model = SimpleBaselineLSTM(
            n_squares=n_squares,
            n_temporal_features=n_temporal_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            prediction_horizon=prediction_horizon
        )

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        print("Baseline model loaded successfully!")
        return model

    def _generate_predictions(self):
        """Generate predictions for validation and test sets"""
        print("Generating predictions...")

        results = {}

        # Process validation set
        val_pred, val_actual, val_days = self._predict_dataset(
            self.data['X_traffic_val'],
            self.data['X_temporal_val'],
            self.data['y_val']
        )

        results['validation'] = {
            'predictions': val_pred,
            'actual': val_actual,
            'days': val_days
        }

        # Process test set
        test_pred, test_actual, test_days = self._predict_dataset(
            self.data['X_traffic_test'],
            self.data['X_temporal_test'],
            self.data['y_test']
        )

        results['test'] = {
            'predictions': test_pred,
            'actual': test_actual,
            'days': test_days
        }

        print("Predictions generated successfully!")
        return results

    def _predict_dataset(self, X_traffic, X_temporal, y_true, batch_size=16):
        """Generate predictions for a dataset"""

        predictions = []
        days = []

        # Create batches
        n_samples = len(X_traffic)
        n_batches = (n_samples + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)

                # Get batch
                batch_traffic = torch.FloatTensor(X_traffic[start_idx:end_idx]).to(self.device)
                batch_temporal = torch.FloatTensor(X_temporal[start_idx:end_idx]).to(self.device)

                # Forward pass
                pred, day_of_week = self.model(batch_traffic, batch_temporal)

                predictions.append(pred.cpu().numpy())
                days.append(day_of_week.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        days = np.concatenate(days, axis=0)

        return predictions, y_true, days

    def calculate_metrics(self, predictions, actual, split_name):
        """Calculate comprehensive metrics"""

        # Flatten for overall metrics
        pred_flat = predictions.reshape(-1)
        actual_flat = actual.reshape(-1)

        # Overall metrics
        mae = mean_absolute_error(actual_flat, pred_flat)
        rmse = np.sqrt(mean_squared_error(actual_flat, pred_flat))
        r2 = r2_score(actual_flat, pred_flat)

        # MAPE with small epsilon for normalized data
        epsilon = 1e-3
        mape = np.mean(np.abs((actual_flat - pred_flat) / (actual_flat + epsilon))) * 100

        # Correlation
        correlation, _ = pearsonr(actual_flat, pred_flat)

        # Overprediction rate
        overpred_rate = np.mean(pred_flat > actual_flat) * 100

        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
            'correlation': float(correlation),
            'overprediction_rate': float(overpred_rate),
            'samples': len(pred_flat)
        }

        print(f"\n{split_name} Metrics:")
        for metric, value in metrics.items():
            if metric != 'samples':
                print(f"  {metric.upper()}: {value:.4f}")
            else:
                print(f"  {metric.upper()}: {value}")

        return metrics

    def create_timeseries_plots(self):
        """Create comprehensive timeseries plots with larger fonts"""
        print("Creating timeseries plots...")

        # TEMPORARY LARGER FONTS FOR THIS FIGURE ONLY
        original_font_config = FONT_CONFIG.copy()
        FONT_CONFIG.update({
            'title': 28,
            'label': 30,
            'tick': 26,
            'legend': 22,
            'day_label': 22
        })

        fig, axes = plt.subplots(2, 1, figsize=FIGURE_SIZES['timeseries_full'])

        # Define day names
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_abbrev = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        for idx, (split_name, ax) in enumerate([('validation', axes[0]), ('test', axes[1])]):
            data = self.results[split_name]
            predictions = data['predictions']
            actual = data['actual']
            days = data['days']

            # Calculate metrics for title
            metrics = self.calculate_metrics(predictions, actual, split_name.capitalize())

            # Average across all squares for visualization
            pred_avg = np.mean(predictions, axis=2)  # Average across squares
            actual_avg = np.mean(actual, axis=2)

            # For visualization, take the first prediction horizon for each sample
            pred_series = pred_avg[:, 0]  # First horizon only
            actual_series = actual_avg[:, 0]

            # Create time index
            time_points = np.arange(len(pred_series))

            # Plot actual and predicted with consistent colors
            ax.plot(time_points, actual_series, color=COLOR_CONFIG['actual'], linewidth=2.5,
                    label='Actual', alpha=0.9)
            ax.plot(time_points, pred_series, color=COLOR_CONFIG['predicted'], linewidth=2,
                    linestyle='--', label='Predicted', alpha=0.9)

            # Add day labels and boundaries
            day_boundaries = []
            day_labels = []
            current_day = days[0]
            boundary_start = 0

            for i, day in enumerate(days):
                if day != current_day or i == len(days) - 1:
                    if i == len(days) - 1:
                        boundary_end = i
                    else:
                        boundary_end = i - 1

                    # Add vertical line at day boundary
                    if i < len(days) - 1:
                        ax.axvline(x=i, color=COLOR_CONFIG['boundary'], linestyle=':',
                                   alpha=0.6, linewidth=1)

                    # Add day label at center of day
                    center_pos = (boundary_start + boundary_end) / 2
                    day_labels.append((center_pos, day_names[current_day]))

                    boundary_start = i
                    current_day = day

            # Add day labels at the top with larger font
            for pos, day_name in day_labels:
                ax.text(pos, ax.get_ylim()[1] * 0.95, day_name,
                        ha='center', va='top', fontweight='bold', fontsize=FONT_CONFIG['day_label'],
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=COLOR_CONFIG['highlight'], alpha=0.7))

            # Formatting with larger fonts
            ax.set_title(f'Baseline Model - {split_name.capitalize()} Set: True vs Predicted Traffic\n'
                         f'MAE: {metrics["mae"]:.6f}, RMSE: {metrics["rmse"]:.6f}, '
                         f'R²: {metrics["r2"]:.4f}, Correlation: {metrics["correlation"]:.4f}',
                         fontsize=FONT_CONFIG['title'], fontweight='bold', pad=20)

            ax.set_xlabel('Time Steps (10-minute intervals)', fontsize=FONT_CONFIG['label'])
            ax.set_ylabel('Average Traffic (Normalized)', fontsize=FONT_CONFIG['label'])
            ax.grid(True, alpha=COLOR_CONFIG['grid_alpha'])

            # Place legend inside the plot with larger font
            ax.legend(loc='upper right', fontsize=FONT_CONFIG['legend'], framealpha=0.9)

            # Improve axis tick formatting
            ax.tick_params(axis='both', which='major', labelsize=FONT_CONFIG['tick'])

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'baseline_timeseries_comparison_full.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()


        # Create detailed view for single day
        self._create_detailed_timeseries()
        FONT_CONFIG.update(original_font_config)

    def _create_detailed_timeseries(self):
        """Create detailed timeseries plot for single day with larger fonts"""
        print("Creating detailed timeseries plots...")

        # TEMPORARY LARGER FONTS FOR THIS FIGURE ONLY
        original_font_config = FONT_CONFIG.copy()
        FONT_CONFIG.update({
            'title': 28,
            'label': 30,
            'tick': 26,
            'legend': 22,
            'day_label': 22
        })

        fig, axes = plt.subplots(2, 1, figsize=FIGURE_SIZES['timeseries_full'])

        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        for idx, (split_name, ax) in enumerate([('validation', axes[0]), ('test', axes[1])]):
            data = self.results[split_name]
            predictions = data['predictions']
            actual = data['actual']
            days = data['days']

            # Find first complete day (144 samples = 1 day)
            samples_per_day = 144
            if len(predictions) < samples_per_day:
                samples_per_day = len(predictions)

            # Take first day worth of data
            pred_subset = predictions[:samples_per_day]
            actual_subset = actual[:samples_per_day]
            days_subset = days[:samples_per_day]

            # Average across squares and horizons for cleaner visualization
            pred_avg = np.mean(pred_subset, axis=(1, 2))  # Average across horizons and squares
            actual_avg = np.mean(actual_subset, axis=(1, 2))

            # Create time index for 24 hours (144 * 10-minute intervals)
            time_points = np.arange(len(pred_avg))

            # Convert time steps to hours for better x-axis labels
            hours = time_points * 10 / 60  # Convert 10-minute intervals to hours

            # Plot with improved styling and consistent colors
            ax.plot(hours, actual_avg, color=COLOR_CONFIG['actual'], linewidth=2.5,
                    label='Actual', alpha=0.9)
            ax.plot(hours, pred_avg, color=COLOR_CONFIG['predicted'], linewidth=2,
                    linestyle='--', label='Predicted', alpha=0.9)

            # Add day name to title
            day_name = day_names[days_subset[0]] if len(days_subset) > 0 else "Unknown"

            # Calculate metrics for this day
            day_mae = mean_absolute_error(actual_avg, pred_avg)
            day_rmse = np.sqrt(mean_squared_error(actual_avg, pred_avg))
            day_r2 = r2_score(actual_avg, pred_avg)

            ax.set_title(f'Baseline Model - {split_name.capitalize()} Set: Single Day Detail ({day_name})\n'
                         f'MAE: {day_mae:.6f}, RMSE: {day_rmse:.6f}, R²: {day_r2:.4f}',
                         fontsize=FONT_CONFIG['title'], fontweight='bold', pad=15)

            ax.set_xlabel('Hour of Day', fontsize=FONT_CONFIG['label'])
            ax.set_ylabel('Average Traffic (Normalized)', fontsize=FONT_CONFIG['label'])

            # Improve x-axis formatting - show every 4 hours
            ax.set_xticks(np.arange(0, 25, 4))
            ax.set_xticklabels([f'{int(h):02d}:00' for h in np.arange(0, 25, 4)])

            # Add vertical lines at key hours (6 AM, 12 PM, 6 PM)
            key_hours = [6, 12, 18]
            for hour in key_hours:
                ax.axvline(x=hour, color=COLOR_CONFIG['boundary'], linestyle=':', alpha=0.5, linewidth=1)
                ax.text(hour, ax.get_ylim()[1] * 0.95, f'{hour:02d}:00',
                        ha='center', va='top', fontsize=FONT_CONFIG['annotation'], alpha=0.7)

            # Place legend inside the plot with larger font
            ax.legend(loc='upper right', fontsize=FONT_CONFIG['legend'], framealpha=0.9,
                      bbox_to_anchor=(0.98, 0.98))

            ax.grid(True, alpha=COLOR_CONFIG['grid_alpha'])
            ax.tick_params(axis='both', which='major', labelsize=FONT_CONFIG['tick'])

            # Set y-axis limits for better visualization
            y_min = min(actual_avg.min(), pred_avg.min()) * 0.95
            y_max = max(actual_avg.max(), pred_avg.max()) * 1.05
            ax.set_ylim(y_min, y_max)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'baseline_timeseries_detailed_view.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()
        FONT_CONFIG.update(original_font_config)

    def create_scatter_plots(self):
        """Create scatter plots for correlation analysis with larger fonts"""
        print("Creating scatter plots...")

        fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES['scatter'])

        for idx, split_name in enumerate(['validation', 'test']):
            data = self.results[split_name]
            predictions = data['predictions'].reshape(-1)
            actual = data['actual'].reshape(-1)

            # Sample data if too large
            if len(predictions) > 10000:
                sample_idx = np.random.choice(len(predictions), 10000, replace=False)
                predictions = predictions[sample_idx]
                actual = actual[sample_idx]

            axes[idx].scatter(actual, predictions, alpha=0.5, s=1, color=COLOR_CONFIG['scatter'])

            # Add perfect prediction line
            min_val = min(actual.min(), predictions.min())
            max_val = max(actual.max(), predictions.max())
            axes[idx].plot([min_val, max_val], [min_val, max_val],
                           color=COLOR_CONFIG['reference'], linestyle='--',
                           linewidth=2, label='Perfect Prediction')

            # Calculate and display R²
            r2 = r2_score(actual, predictions)
            correlation, _ = pearsonr(actual, predictions)

            axes[idx].set_title(f'Baseline Model - {split_name.capitalize()} Set Correlation\n'
                                f'R² = {r2:.4f}, Correlation = {correlation:.4f}',
                                fontsize=FONT_CONFIG['title'], fontweight='bold')
            axes[idx].set_xlabel('Actual Traffic', fontsize=FONT_CONFIG['label'])
            axes[idx].set_ylabel('Predicted Traffic', fontsize=FONT_CONFIG['label'])
            axes[idx].legend(fontsize=FONT_CONFIG['legend'])
            axes[idx].grid(True, alpha=COLOR_CONFIG['grid_alpha'])
            axes[idx].tick_params(axis='both', which='major', labelsize=FONT_CONFIG['tick'])

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'baseline_correlation_scatter_plots.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

    def create_day_specific_analysis(self):
        """Create day-specific performance analysis with larger fonts"""
        print("Creating day-specific analysis...")

        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        # Collect day-specific metrics
        day_metrics = {'validation': {}, 'test': {}}

        for split_name in ['validation', 'test']:
            data = self.results[split_name]
            predictions = data['predictions']
            actual = data['actual']
            days = data['days']

            for day in range(7):
                day_mask = (days == day)
                if np.sum(day_mask) > 0:
                    day_pred = predictions[day_mask].reshape(-1)
                    day_actual = actual[day_mask].reshape(-1)

                    mae = mean_absolute_error(day_actual, day_pred)
                    rmse = np.sqrt(mean_squared_error(day_actual, day_pred))
                    r2 = r2_score(day_actual, day_pred)
                    correlation, _ = pearsonr(day_actual, day_pred)

                    day_metrics[split_name][day_names[day]] = {
                        'mae': mae,
                        'rmse': rmse,
                        'r2': r2,
                        'correlation': correlation,
                        'samples': np.sum(day_mask)
                    }

        # Create visualization with larger fonts
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZES['day_analysis'])

        metrics_to_plot = ['mae', 'rmse', 'r2', 'correlation']
        metric_titles = ['MAE', 'RMSE', 'R²', 'Correlation']

        for i, (metric, title) in enumerate(zip(metrics_to_plot, metric_titles)):
            ax = axes[i // 2, i % 2]

            val_values = [day_metrics['validation'][day][metric] for day in day_names
                          if day in day_metrics['validation']]
            test_values = [day_metrics['test'][day][metric] for day in day_names
                           if day in day_metrics['test']]

            x = np.arange(len(day_names))
            width = 0.35

            ax.bar(x - width / 2, val_values, width, label='Validation',
                   alpha=0.8, color=COLOR_CONFIG['validation'])
            ax.bar(x + width / 2, test_values, width, label='Test',
                   alpha=0.8, color=COLOR_CONFIG['test'])

            ax.set_title(f'Baseline Model - {title} by Day of Week',
                         fontsize=FONT_CONFIG['subtitle'], fontweight='bold')
            ax.set_xlabel('Day of Week', fontsize=FONT_CONFIG['label'])
            ax.set_ylabel(title, fontsize=FONT_CONFIG['label'])
            ax.set_xticks(x)
            ax.set_xticklabels([day[:3] for day in day_names])
            ax.legend(fontsize=FONT_CONFIG['legend'])
            ax.grid(True, alpha=COLOR_CONFIG['grid_alpha'])
            ax.tick_params(axis='both', which='major', labelsize=FONT_CONFIG['tick'])

        plt.suptitle('Baseline Model - Day-Specific Performance Analysis',
                     fontsize=FONT_CONFIG['title'], fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'baseline_day_specific_performance.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

        return day_metrics

    def create_error_analysis(self):
        """Create error distribution and analysis plots with larger fonts"""
        print("Creating error analysis...")

        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZES['error_analysis'])

        for idx, split_name in enumerate(['validation', 'test']):
            data = self.results[split_name]
            predictions = data['predictions'].reshape(-1)
            actual = data['actual'].reshape(-1)
            errors = predictions - actual

            # Error histogram
            axes[0, idx].hist(errors, bins=100, alpha=0.7, density=True, color=COLOR_CONFIG['validation'])
            axes[0, idx].set_title(f'Baseline Model - {split_name.capitalize()} Error Distribution',
                                   fontsize=FONT_CONFIG['subtitle'], fontweight='bold')
            axes[0, idx].set_xlabel('Prediction Error', fontsize=FONT_CONFIG['label'])
            axes[0, idx].set_ylabel('Density', fontsize=FONT_CONFIG['label'])
            axes[0, idx].axvline(x=0, color=COLOR_CONFIG['reference'], linestyle='--', alpha=0.7)
            axes[0, idx].grid(True, alpha=COLOR_CONFIG['grid_alpha'])
            axes[0, idx].tick_params(axis='both', which='major', labelsize=FONT_CONFIG['tick'])

            # Add statistics to plot
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            axes[0, idx].text(0.05, 0.95, f'Mean: {mean_error:.4f}\nStd: {std_error:.4f}',
                              transform=axes[0, idx].transAxes, va='top',
                              fontsize=FONT_CONFIG['annotation'],
                              bbox=dict(boxstyle="round,pad=0.3", facecolor=COLOR_CONFIG['highlight']))

            # Error vs actual values
            sample_idx = np.random.choice(len(errors), min(5000, len(errors)), replace=False)
            axes[1, idx].scatter(actual[sample_idx], errors[sample_idx], alpha=0.5, s=1,
                                 color=COLOR_CONFIG['error_scatter'])
            axes[1, idx].set_title(f'Baseline Model - {split_name.capitalize()} Error vs Actual',
                                   fontsize=FONT_CONFIG['subtitle'], fontweight='bold')
            axes[1, idx].set_xlabel('Actual Traffic', fontsize=FONT_CONFIG['label'])
            axes[1, idx].set_ylabel('Prediction Error', fontsize=FONT_CONFIG['label'])
            axes[1, idx].axhline(y=0, color=COLOR_CONFIG['reference'], linestyle='--', alpha=0.7)
            axes[1, idx].grid(True, alpha=COLOR_CONFIG['grid_alpha'])
            axes[1, idx].tick_params(axis='both', which='major', labelsize=FONT_CONFIG['tick'])

        plt.suptitle('Baseline Model - Error Analysis', fontsize=FONT_CONFIG['title'],
                     fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'baseline_error_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

    def create_horizon_analysis(self):
        """Analyze performance across prediction horizons with larger fonts"""
        print("Creating horizon analysis...")

        horizon_metrics = {'validation': [], 'test': []}

        for split_name in ['validation', 'test']:
            data = self.results[split_name]
            predictions = data['predictions']
            actual = data['actual']

            n_horizons = predictions.shape[1]

            for h in range(n_horizons):
                pred_h = predictions[:, h, :].reshape(-1)
                actual_h = actual[:, h, :].reshape(-1)

                mae = mean_absolute_error(actual_h, pred_h)
                rmse = np.sqrt(mean_squared_error(actual_h, pred_h))
                r2 = r2_score(actual_h, pred_h)

                horizon_metrics[split_name].append({
                    'horizon': h + 1,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                })

        # Plot horizon analysis with larger fonts
        fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES['horizon_analysis'])
        metrics = ['mae', 'rmse', 'r2']
        titles = ['MAE vs Horizon', 'RMSE vs Horizon', 'R² vs Horizon']

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            val_values = [m[metric] for m in horizon_metrics['validation']]
            test_values = [m[metric] for m in horizon_metrics['test']]
            horizons = [m['horizon'] for m in horizon_metrics['validation']]

            axes[i].plot(horizons, val_values, 'o-', label='Validation', linewidth=2,
                         color=COLOR_CONFIG['validation'])
            axes[i].plot(horizons, test_values, 's-', label='Test', linewidth=2,
                         color=COLOR_CONFIG['test'])
            axes[i].set_title(f'Baseline Model - {title}', fontsize=FONT_CONFIG['subtitle'],
                              fontweight='bold')
            axes[i].set_xlabel('Prediction Horizon (10-min intervals)', fontsize=FONT_CONFIG['label'])
            axes[i].set_ylabel(metric.upper(), fontsize=FONT_CONFIG['label'])
            axes[i].legend(fontsize=FONT_CONFIG['legend'])
            axes[i].grid(True, alpha=COLOR_CONFIG['grid_alpha'])
            axes[i].tick_params(axis='both', which='major', labelsize=FONT_CONFIG['tick'])

        plt.suptitle('Baseline Model - Horizon Analysis', fontsize=FONT_CONFIG['title'],
                     fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'baseline_horizon_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

        return horizon_metrics

    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

    def _create_summary_table(self, val_metrics, test_metrics):
        """Create a summary performance table with larger fonts"""

        fig, ax = plt.subplots(figsize=FIGURE_SIZES['summary_table'])
        ax.axis('tight')
        ax.axis('off')

        # Helper function to calculate improvement correctly based on metric type
        def calculate_improvement(val_value, test_value, metric_name):
            if metric_name in ['mae', 'rmse', 'mape', 'overprediction_rate']:
                # For these metrics, lower is better, so improvement = (val - test) / val * 100
                return ((val_value - test_value) / val_value * 100)
            else:
                # For R² and correlation, higher is better, so improvement = test - val
                return (test_value - val_value)

        # Create table data with corrected improvement calculations
        table_data = [
            ['Metric', 'Validation', 'Test', 'Val vs Test'],
            ['MAE', f"{val_metrics['mae']:.6f}", f"{test_metrics['mae']:.6f}",
             f"{calculate_improvement(val_metrics['mae'], test_metrics['mae'], 'mae'):+.2f}%"],
            ['RMSE', f"{val_metrics['rmse']:.6f}", f"{test_metrics['rmse']:.6f}",
             f"{calculate_improvement(val_metrics['rmse'], test_metrics['rmse'], 'rmse'):+.2f}%"],
            ['R²', f"{val_metrics['r2']:.6f}", f"{test_metrics['r2']:.6f}",
             f"{calculate_improvement(val_metrics['r2'], test_metrics['r2'], 'r2'):+.6f}"],
            ['MAPE (%)', f"{val_metrics['mape']:.2f}%", f"{test_metrics['mape']:.2f}%",
             f"{calculate_improvement(val_metrics['mape'], test_metrics['mape'], 'mape'):+.2f}%"],
            ['Correlation', f"{val_metrics['correlation']:.6f}", f"{test_metrics['correlation']:.6f}",
             f"{calculate_improvement(val_metrics['correlation'], test_metrics['correlation'], 'correlation'):+.6f}"],
            ['Overpred Rate (%)', f"{val_metrics['overprediction_rate']:.2f}%",
             f"{test_metrics['overprediction_rate']:.2f}%",
             f"{calculate_improvement(val_metrics['overprediction_rate'], test_metrics['overprediction_rate'], 'overprediction_rate'):+.2f}%"]
        ]

        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(FONT_CONFIG['table'])
        table.scale(1, 2)

        # Style the table
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor(COLOR_CONFIG['header'])
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color code performance cells
        for i in range(1, len(table_data)):
            for j in range(1, 3):  # Validation and Test columns
                if 'R²' in table_data[i][0]:
                    value = float(table_data[i][j])
                    if value >= 0.7:
                        table[(i, j)].set_facecolor('#90EE90')  # Light green
                    elif value >= 0.5:
                        table[(i, j)].set_facecolor('#FFFFE0')  # Light yellow
                    else:
                        table[(i, j)].set_facecolor('#FFB6C1')  # Light pink

        # Color code the improvement column based on whether improvement is positive
        for i in range(1, len(table_data)):
            improvement_text = table_data[i][3]
            # Extract the numeric value (remove + or - and % signs)
            if '%' in improvement_text:
                improvement_value = float(improvement_text.replace('%', '').replace('+', ''))
            else:
                improvement_value = float(improvement_text.replace('+', ''))

            # Color based on whether improvement is positive (green) or negative (light red)
            metric_name = table_data[i][0].lower()
            if any(m in metric_name for m in ['mae', 'rmse', 'mape', 'overpred']):
                # Lower is better - green for negative improvement
                if improvement_value < 0:
                    table[(i, 3)].set_facecolor('#90EE90')
                else:
                    table[(i, 3)].set_facecolor('#FFB6C1')
            else:
                # Higher is better - green for positive improvement
                if improvement_value > 0:
                    table[(i, 3)].set_facecolor('#90EE90')
                else:
                    table[(i, 3)].set_facecolor('#FFB6C1')

        plt.title('Baseline Model Performance Summary', fontsize=FONT_CONFIG['title'],
                  fontweight='bold', pad=20)
        plt.savefig(os.path.join(self.output_dir, 'baseline_performance_summary_table.png'),
                    dpi=300, bbox_inches='tight')
        plt.show()

    def generate_comprehensive_report(self):
        """Generate comprehensive evaluation report"""
        print("Generating comprehensive report...")

        # Calculate overall metrics
        val_metrics = self.calculate_metrics(
            self.results['validation']['predictions'],
            self.results['validation']['actual'],
            'Validation'
        )

        test_metrics = self.calculate_metrics(
            self.results['test']['predictions'],
            self.results['test']['actual'],
            'Test'
        )

        # Generate all visualizations
        self.create_timeseries_plots()
        self.create_scatter_plots()
        day_metrics = self.create_day_specific_analysis()
        self.create_error_analysis()
        horizon_metrics = self.create_horizon_analysis()

        # Create summary report
        report = {
            'model_type': 'Baseline CNN-LSTM',
            'overall_metrics': {
                'validation': val_metrics,
                'test': test_metrics
            },
            'day_specific_metrics': day_metrics,
            'horizon_metrics': horizon_metrics,
            'model_info': {
                'data_path': self.data_path,
                'model_path': self.model_path,
                'evaluation_date': datetime.now().isoformat()
            }
        }

        # Convert numpy types to native Python types
        report = self._convert_numpy_types(report)

        # Save report as JSON
        import json
        try:
            with open(os.path.join(self.output_dir, 'baseline_evaluation_report.json'), 'w') as f:
                json.dump(report, f, indent=2)
            print("Baseline evaluation report saved successfully!")
        except Exception as e:
            print(f"Warning: Could not save JSON report due to: {e}")
            # Save as pickle instead
            import pickle
            with open(os.path.join(self.output_dir, 'baseline_evaluation_report.pkl'), 'wb') as f:
                pickle.dump(report, f)
            print("Report saved as pickle file instead.")

        # Create summary table
        self._create_summary_table(val_metrics, test_metrics)

        print(f"\nComprehensive baseline evaluation completed!")
        print(f"Results saved to: {self.output_dir}")
        print(f"Key files generated:")
        print(f"  - baseline_timeseries_comparison_full.png")
        print(f"  - baseline_timeseries_detailed_view.png")
        print(f"  - baseline_correlation_scatter_plots.png")
        print(f"  - baseline_day_specific_performance.png")
        print(f"  - baseline_error_analysis.png")
        print(f"  - baseline_horizon_analysis.png")
        print(f"  - baseline_evaluation_report.json (or .pkl)")
        print(f"  - baseline_performance_summary_table.png")

        return report


def main():
    """Main execution function"""
    print("=" * 80)
    print("COMPREHENSIVE BASELINE TRAFFIC PREDICTION MODEL EVALUATION")
    print("=" * 80)

    # File paths - Updated for your baseline model
    data_path = r"C:\Users\Fan\PycharmProjects\Msc_thesis_final\processed_data\preprocessed_milan_traffic_center_1600_7day_splits_optimized.pkl"
    model_path = r"C:\Users\Fan\PycharmProjects\Msc_Thesis\baseline_model\baseline_outputs\baseline_simple_cnn_lstm_20250831_162807\models\best_baseline.pth"

    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"baseline_evaluation_results_{timestamp}"

    # Check if files exist
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Trying alternative data path...")
        alt_data_path = r"C:\Users\Fan\PycharmProjects\Msc_Thesis\processed_data\preprocessed_milan_traffic_center_1600_7day_splits_optimized.pkl"
        if os.path.exists(alt_data_path):
            data_path = alt_data_path
            print(f"Using alternative data path: {data_path}")
        else:
            print(f"Error: Data file not found at alternative path either: {alt_data_path}")
            return

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    try:
        # Create evaluator
        evaluator = BaselineModelEvaluator(
            data_path=data_path,
            model_path=model_path,
            output_dir=output_dir
        )

        # Generate comprehensive evaluation
        report = evaluator.generate_comprehensive_report()

        # Print summary of baseline performance
        val_metrics = report['overall_metrics']['validation']
        test_metrics = report['overall_metrics']['test']

        print("\n" + "=" * 80)
        print("BASELINE MODEL EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Baseline Performance Summary:")
        print(f"  Validation R²: {val_metrics['r2']:.4f}")
        print(f"  Test R²: {test_metrics['r2']:.4f}")
        print(f"  Validation MAE: {val_metrics['mae']:.6f}")
        print(f"  Test MAE: {test_metrics['mae']:.6f}")
        print(f"  Generalization Gap (R²): {abs(val_metrics['r2'] - test_metrics['r2']):.4f}")

        # Performance assessment
        if test_metrics['r2'] >= 0.7:
            print(f"  🎯 Excellent baseline! R² ≥ 0.7 achieved")
        elif test_metrics['r2'] >= 0.5:
            print(f"  ✅ Strong baseline performance (R² ≥ 0.5)")
        elif test_metrics['r2'] >= 0.3:
            print(f"  ⚠️ Moderate baseline performance")
        else:
            print(f"  ❌ Weak baseline - consider model improvements")

        print(f"\nGenerated comprehensive visualizations for thesis:")
        print(f"✓ Full timeseries comparison with day labels")
        print(f"✓ Single-day detailed analysis")
        print(f"✓ Correlation scatter plots")
        print(f"✓ Day-specific performance breakdown")
        print(f"✓ Error distribution analysis")
        print(f"✓ Prediction horizon performance")
        print(f"✓ Performance summary table")

        print(f"\nAll results saved to: '{output_dir}' folder")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

        print("\nTroubleshooting suggestions:")
        print("1. Verify that the baseline model checkpoint contains all required information")
        print("2. Check that the data file format matches expectations")
        print("3. Ensure sufficient memory is available")
        print("4. Try running on a machine with CUDA if available for faster processing")


if __name__ == "__main__":
    main()