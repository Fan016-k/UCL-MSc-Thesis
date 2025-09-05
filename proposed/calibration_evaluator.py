#!/usr/bin/env python3
"""
Comprehensive Calibration Evaluation for Traffic Prediction Models
Implements OPR, UPR, CR, SB metrics with macro/micro averages and peak-weighted variants
"""
import os
# Windows-specific fixes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads
import torch
import torch.nn as nn
import numpy as np
import pickle
import json
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class CalibrationEvaluator:
    """
    Comprehensive calibration evaluator for traffic prediction models
    """

    def __init__(self, tolerance_delta=0.01, peak_quantiles=[0.80, 0.95]):
        """
        Initialize calibration evaluator

        Args:
            tolerance_delta: Tolerance threshold for calibration (default 0.01 for [0,1] scaled data)
            peak_quantiles: Quantiles to define peak traffic periods
        """
        self.delta = tolerance_delta
        self.peak_quantiles = peak_quantiles
        self.results = {}

    def compute_horizon_metrics(self, predictions, targets, horizon_idx):
        """
        Compute calibration metrics for a specific prediction horizon

        Args:
            predictions: Array of predictions for this horizon [N_samples]
            targets: Array of true values for this horizon [N_samples]
            horizon_idx: Horizon index (0-based)

        Returns:
            Dictionary with OPR, UPR, CR, SB metrics
        """
        # Signed errors: e = prediction - target
        signed_errors = predictions - targets
        N_h = len(signed_errors)

        if N_h == 0:
            return {
                'OPR': 0.0, 'UPR': 0.0, 'CR': 1.0, 'SB': 0.0,
                'samples': 0, 'horizon': horizon_idx
            }

        # Over-prediction rate: proportion where error > delta
        OPR = np.mean(signed_errors > self.delta)

        # Under-prediction rate: proportion where error < -delta
        UPR = np.mean(signed_errors < -self.delta)

        # Calibration rate: proportion within tolerance
        CR = 1.0 - OPR - UPR

        # Sign balance: difference between over and under prediction
        SB = OPR - UPR

        return {
            'OPR': float(OPR),
            'UPR': float(UPR),
            'CR': float(CR),
            'SB': float(SB),
            'samples': int(N_h),
            'horizon': int(horizon_idx)
        }

    def compute_peak_weighted_metrics(self, predictions, targets, horizon_idx):
        """
        Compute peak-weighted calibration metrics for high-traffic periods

        Args:
            predictions: Array of predictions [N_samples]
            targets: Array of true values [N_samples]
            horizon_idx: Horizon index

        Returns:
            Dictionary with peak-weighted metrics for each quantile
        """
        signed_errors = predictions - targets
        peak_metrics = {}

        for q in self.peak_quantiles:
            # Define peak threshold as q-quantile of targets
            peak_threshold = np.quantile(targets, q)

            # Peak indicator: 1 if target >= threshold
            peak_mask = targets >= peak_threshold
            n_peak = np.sum(peak_mask)

            if n_peak == 0:
                peak_metrics[f'q{int(q * 100)}'] = {
                    'OPR_peak': 0.0,
                    'UPR_peak': 0.0,
                    'CR_peak': 1.0,
                    'SB_peak': 0.0,
                    'peak_samples': 0,
                    'peak_threshold': float(peak_threshold)
                }
                continue

            # Peak-weighted over-prediction rate
            OPR_peak = np.sum((signed_errors > self.delta) & peak_mask) / n_peak

            # Peak-weighted under-prediction rate
            UPR_peak = np.sum((signed_errors < -self.delta) & peak_mask) / n_peak

            # Peak calibration rate
            CR_peak = 1.0 - OPR_peak - UPR_peak

            # Peak sign balance
            SB_peak = OPR_peak - UPR_peak

            peak_metrics[f'q{int(q * 100)}'] = {
                'OPR_peak': float(OPR_peak),
                'UPR_peak': float(UPR_peak),
                'CR_peak': float(CR_peak),
                'SB_peak': float(SB_peak),
                'peak_samples': int(n_peak),
                'peak_threshold': float(peak_threshold)
            }

        return peak_metrics

    def compute_macro_averages(self, horizon_metrics):
        """
        Compute macro-averaged metrics (treat all horizons equally)

        Args:
            horizon_metrics: List of horizon-specific metric dictionaries

        Returns:
            Dictionary with macro-averaged metrics
        """
        if not horizon_metrics:
            return {'OPR_macro': 0.0, 'UPR_macro': 0.0, 'CR_macro': 1.0, 'SB_macro': 0.0}

        H = len(horizon_metrics)

        OPR_macro = np.mean([h['OPR'] for h in horizon_metrics])
        UPR_macro = np.mean([h['UPR'] for h in horizon_metrics])
        CR_macro = np.mean([h['CR'] for h in horizon_metrics])
        SB_macro = np.mean([h['SB'] for h in horizon_metrics])

        return {
            'OPR_macro': float(OPR_macro),
            'UPR_macro': float(UPR_macro),
            'CR_macro': float(CR_macro),
            'SB_macro': float(SB_macro),
            'horizons': H
        }

    def compute_micro_averages(self, predictions_all, targets_all):
        """
        Compute micro-averaged metrics (weight all instances equally)

        Args:
            predictions_all: Flattened predictions across all horizons [N_total]
            targets_all: Flattened targets across all horizons [N_total]

        Returns:
            Dictionary with micro-averaged metrics
        """
        signed_errors = predictions_all - targets_all
        N_total = len(signed_errors)

        if N_total == 0:
            return {'OPR_micro': 0.0, 'UPR_micro': 0.0, 'CR_micro': 1.0, 'SB_micro': 0.0}

        OPR_micro = np.mean(signed_errors > self.delta)
        UPR_micro = np.mean(signed_errors < -self.delta)
        CR_micro = 1.0 - OPR_micro - UPR_micro
        SB_micro = OPR_micro - UPR_micro

        return {
            'OPR_micro': float(OPR_micro),
            'UPR_micro': float(UPR_micro),
            'CR_micro': float(CR_micro),
            'SB_micro': float(SB_micro),
            'total_samples': int(N_total)
        }

    def evaluate_model_calibration(self, predictions, targets, model_name="model"):
        """
        Complete calibration evaluation for a model

        Args:
            predictions: Model predictions [N_samples, H_horizons, N_squares]
            targets: True values [N_samples, H_horizons, N_squares]
            model_name: Name identifier for the model

        Returns:
            Comprehensive calibration results dictionary
        """
        print(f"\nEvaluating calibration for {model_name}...")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Tolerance delta: {self.delta}")

        N_samples, H_horizons, N_squares = predictions.shape

        # Initialize results structure
        results = {
            'model_name': model_name,
            'tolerance_delta': self.delta,
            'peak_quantiles': self.peak_quantiles,
            'prediction_shape': list(predictions.shape),
            'horizon_metrics': [],
            'horizon_peak_metrics': []
        }

        # Collect all predictions and targets for micro-averaging
        all_predictions = []
        all_targets = []

        # Process each horizon
        for h in range(H_horizons):
            print(f"  Processing horizon {h + 1}/{H_horizons}...")

            # Flatten spatial dimension for this horizon
            pred_h = predictions[:, h, :].reshape(-1)  # [N_samples * N_squares]
            target_h = targets[:, h, :].reshape(-1)

            # Remove any invalid values
            valid_mask = ~(np.isnan(pred_h) | np.isnan(target_h) |
                           np.isinf(pred_h) | np.isinf(target_h))

            pred_h_valid = pred_h[valid_mask]
            target_h_valid = target_h[valid_mask]

            if len(pred_h_valid) == 0:
                print(f"    Warning: No valid samples for horizon {h}")
                continue

            # Compute horizon-specific metrics
            horizon_metrics = self.compute_horizon_metrics(pred_h_valid, target_h_valid, h)
            results['horizon_metrics'].append(horizon_metrics)

            # Compute peak-weighted metrics
            peak_metrics = self.compute_peak_weighted_metrics(pred_h_valid, target_h_valid, h)
            peak_metrics['horizon'] = h
            results['horizon_peak_metrics'].append(peak_metrics)

            # Accumulate for micro-averaging
            all_predictions.extend(pred_h_valid)
            all_targets.extend(target_h_valid)

            print(f"    Horizon {h}: OPR={horizon_metrics['OPR']:.3f}, "
                  f"UPR={horizon_metrics['UPR']:.3f}, CR={horizon_metrics['CR']:.3f}, "
                  f"SB={horizon_metrics['SB']:.3f}")

        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # Compute macro averages
        macro_metrics = self.compute_macro_averages(results['horizon_metrics'])
        results['macro_averages'] = macro_metrics

        # Compute micro averages
        micro_metrics = self.compute_micro_averages(all_predictions, all_targets)
        results['micro_averages'] = micro_metrics

        # Summary statistics
        results['summary'] = {
            'valid_horizons': len(results['horizon_metrics']),
            'total_valid_samples': int(len(all_predictions)),
            'mean_horizon_samples': np.mean([h['samples'] for h in results['horizon_metrics']]) if results[
                'horizon_metrics'] else 0
        }

        print(f"Calibration evaluation complete for {model_name}")
        print(f"  Macro averages: OPR={macro_metrics['OPR_macro']:.3f}, "
              f"UPR={macro_metrics['UPR_macro']:.3f}, CR={macro_metrics['CR_macro']:.3f}")
        print(f"  Micro averages: OPR={micro_metrics['OPR_micro']:.3f}, "
              f"UPR={micro_metrics['UPR_micro']:.3f}, CR={micro_metrics['CR_micro']:.3f}")

        return results


class ModelLoader:
    """
    Utility class to load and run different model architectures
    """

    @staticmethod
    def load_baseline_model(model_path, n_squares=1600, n_temporal_features=11):
        """Load baseline model"""
        print(f"Loading baseline model from: {model_path}")

        checkpoint = torch.load(model_path, map_location='cpu')
        config_dict = checkpoint.get('config', {})

        # Import baseline model class
        from baseline_model import SimpleBaselineLSTM

        model = SimpleBaselineLSTM(
            n_squares=n_squares,
            n_temporal_features=n_temporal_features,
            hidden_dim=config_dict.get('HIDDEN_DIM', 256),
            num_layers=config_dict.get('NUM_LAYERS', 2),
            prediction_horizon=config_dict.get('PREDICTION_HORIZON', 12)
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f"Baseline model loaded successfully")
        return model

    @staticmethod
    def load_complex_model(model_path, n_squares=1600, n_temporal_features=11):
        """Load complex/proposed model"""
        print(f"Loading complex model from: {model_path}")

        checkpoint = torch.load(model_path, map_location='cpu')
        config_dict = checkpoint.get('config', {})

        # Import complex model class
        from model import create_reasonable_complex_model

        model, _ = create_reasonable_complex_model(
            n_squares, n_temporal_features, use_original_complexity=True
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f"Complex model loaded successfully")
        return model

    @staticmethod
    def generate_predictions(model, X_traffic, X_temporal, batch_size=16, device='cpu'):
        """Generate predictions from model"""

        class SimpleDataset(Dataset):
            def __init__(self, X_traffic, X_temporal):
                self.X_traffic = torch.FloatTensor(X_traffic)
                self.X_temporal = torch.FloatTensor(X_temporal)

            def __len__(self):
                return len(self.X_traffic)

            def __getitem__(self, idx):
                return self.X_traffic[idx], self.X_temporal[idx]

        dataset = SimpleDataset(X_traffic, X_temporal)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        model = model.to(device)
        model.eval()

        all_predictions = []

        with torch.no_grad():
            for x_traffic, x_temporal in dataloader:
                x_traffic = x_traffic.to(device)
                x_temporal = x_temporal.to(device)

                # Handle different model output formats
                output = model(x_traffic, x_temporal)
                if isinstance(output, tuple):
                    predictions = output[0]  # (batch, horizon, squares)
                else:
                    predictions = output

                all_predictions.append(predictions.cpu().numpy())

        return np.concatenate(all_predictions, axis=0)


def run_calibration_comparison(baseline_path, complex_path, data_path, output_dir=None, tolerance_delta=0.01):
    """
    Main function to run calibration comparison between baseline and complex models

    Args:
        baseline_path: Path to baseline model checkpoint
        complex_path: Path to complex model checkpoint
        data_path: Path to preprocessed data file
        output_dir: Directory to save results (optional)
        tolerance_delta: Calibration tolerance threshold
    """

    print("=" * 80)
    print("COMPREHENSIVE CALIBRATION EVALUATION")
    print("=" * 80)

    # Create output directory if needed
    if output_dir is None:
        output_dir = f"calibration_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print(f"Loading data from: {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    n_squares = data['X_traffic_test'].shape[2]
    n_temporal_features = data['X_temporal_test'].shape[2]

    print(f"Data loaded:")
    print(f"  n_squares: {n_squares}")
    print(f"  n_temporal_features: {n_temporal_features}")
    print(f"  Test samples: {data['X_traffic_test'].shape[0]}")

    # Initialize calibration evaluator
    evaluator = CalibrationEvaluator(tolerance_delta=tolerance_delta)

    # Load models
    baseline_model = ModelLoader.load_baseline_model(baseline_path, n_squares, n_temporal_features)
    complex_model = ModelLoader.load_complex_model(complex_path, n_squares, n_temporal_features)

    # Generate predictions on test set
    print("Generating baseline predictions...")
    baseline_predictions = ModelLoader.generate_predictions(
        baseline_model, data['X_traffic_test'], data['X_temporal_test']
    )

    print("Generating complex model predictions...")
    complex_predictions = ModelLoader.generate_predictions(
        complex_model, data['X_traffic_test'], data['X_temporal_test']
    )

    targets = data['y_test']

    print(f"Prediction shapes:")
    print(f"  Baseline: {baseline_predictions.shape}")
    print(f"  Complex: {complex_predictions.shape}")
    print(f"  Targets: {targets.shape}")

    # Ensure consistent shapes
    min_samples = min(baseline_predictions.shape[0], complex_predictions.shape[0], targets.shape[0])
    min_horizons = min(baseline_predictions.shape[1], complex_predictions.shape[1], targets.shape[1])
    min_squares = min(baseline_predictions.shape[2], complex_predictions.shape[2], targets.shape[2])

    baseline_predictions = baseline_predictions[:min_samples, :min_horizons, :min_squares]
    complex_predictions = complex_predictions[:min_samples, :min_horizons, :min_squares]
    targets = targets[:min_samples, :min_horizons, :min_squares]

    print(f"Aligned shapes: {baseline_predictions.shape}")

    # Evaluate calibration for both models
    baseline_results = evaluator.evaluate_model_calibration(
        baseline_predictions, targets, "Baseline_CNN_LSTM"
    )

    complex_results = evaluator.evaluate_model_calibration(
        complex_predictions, targets, "Complex_Proposed"
    )

    # Create comparison summary
    comparison = create_calibration_comparison(baseline_results, complex_results)

    # Save results
    save_calibration_results(baseline_results, complex_results, comparison, output_dir)

    # Generate visualizations
    create_calibration_visualizations(baseline_results, complex_results, output_dir)

    print(f"\nCalibration evaluation complete!")
    print(f"Results saved to: {output_dir}")

    return baseline_results, complex_results, comparison


def create_calibration_comparison(baseline_results, complex_results):
    """Create comparison summary between models"""

    comparison = {
        'models_compared': [baseline_results['model_name'], complex_results['model_name']],
        'tolerance_delta': baseline_results['tolerance_delta'],
        'evaluation_timestamp': datetime.now().isoformat(),
    }

    # Compare macro averages
    b_macro = baseline_results['macro_averages']
    c_macro = complex_results['macro_averages']

    comparison['macro_comparison'] = {
        'OPR': {
            'baseline': b_macro['OPR_macro'],
            'complex': c_macro['OPR_macro'],
            'improvement': b_macro['OPR_macro'] - c_macro['OPR_macro'],  # Lower is better
            'relative_change_pct': ((c_macro['OPR_macro'] - b_macro['OPR_macro']) / max(b_macro['OPR_macro'],
                                                                                        1e-10)) * 100
        },
        'UPR': {
            'baseline': b_macro['UPR_macro'],
            'complex': c_macro['UPR_macro'],
            'improvement': b_macro['UPR_macro'] - c_macro['UPR_macro'],  # Lower is better
            'relative_change_pct': ((c_macro['UPR_macro'] - b_macro['UPR_macro']) / max(b_macro['UPR_macro'],
                                                                                        1e-10)) * 100
        },
        'CR': {
            'baseline': b_macro['CR_macro'],
            'complex': c_macro['CR_macro'],
            'improvement': c_macro['CR_macro'] - b_macro['CR_macro'],  # Higher is better
            'relative_change_pct': ((c_macro['CR_macro'] - b_macro['CR_macro']) / max(b_macro['CR_macro'], 1e-10)) * 100
        },
        'SB_abs': {
            'baseline': abs(b_macro['SB_macro']),
            'complex': abs(c_macro['SB_macro']),
            'improvement': abs(b_macro['SB_macro']) - abs(c_macro['SB_macro']),  # Lower absolute value is better
        }
    }

    # Compare micro averages
    b_micro = baseline_results['micro_averages']
    c_micro = complex_results['micro_averages']

    comparison['micro_comparison'] = {
        'OPR': {
            'baseline': b_micro['OPR_micro'],
            'complex': c_micro['OPR_micro'],
            'improvement': b_micro['OPR_micro'] - c_micro['OPR_micro'],
            'relative_change_pct': ((c_micro['OPR_micro'] - b_micro['OPR_micro']) / max(b_micro['OPR_micro'],
                                                                                        1e-10)) * 100
        },
        'UPR': {
            'baseline': b_micro['UPR_micro'],
            'complex': c_micro['UPR_micro'],
            'improvement': b_micro['UPR_micro'] - c_micro['UPR_micro'],
            'relative_change_pct': ((c_micro['UPR_micro'] - b_micro['UPR_micro']) / max(b_micro['UPR_micro'],
                                                                                        1e-10)) * 100
        },
        'CR': {
            'baseline': b_micro['CR_micro'],
            'complex': c_micro['CR_micro'],
            'improvement': c_micro['CR_micro'] - b_micro['CR_micro'],
            'relative_change_pct': ((c_micro['CR_micro'] - b_micro['CR_micro']) / max(b_micro['CR_micro'], 1e-10)) * 100
        },
        'SB_abs': {
            'baseline': abs(b_micro['SB_micro']),
            'complex': abs(c_micro['SB_micro']),
            'improvement': abs(b_micro['SB_micro']) - abs(c_micro['SB_micro']),
        }
    }

    # Overall assessment
    cr_improvement = comparison['macro_comparison']['CR']['improvement']
    sb_improvement = comparison['macro_comparison']['SB_abs']['improvement']

    if cr_improvement > 0.01 and sb_improvement > 0.001:
        assessment = "SIGNIFICANT_IMPROVEMENT"
    elif cr_improvement > 0.005:
        assessment = "MODERATE_IMPROVEMENT"
    elif cr_improvement > 0:
        assessment = "MARGINAL_IMPROVEMENT"
    else:
        assessment = "NO_IMPROVEMENT"

    comparison['overall_assessment'] = assessment

    return comparison


def save_calibration_results(baseline_results, complex_results, comparison, output_dir):
    """Save all calibration results to JSON files"""

    # Save individual model results
    with open(os.path.join(output_dir, 'baseline_calibration.json'), 'w') as f:
        json.dump(baseline_results, f, indent=2)

    with open(os.path.join(output_dir, 'complex_calibration.json'), 'w') as f:
        json.dump(complex_results, f, indent=2)

    # Save comparison
    with open(os.path.join(output_dir, 'calibration_comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)

    # Create summary report
    create_calibration_report(baseline_results, complex_results, comparison, output_dir)


def create_calibration_report(baseline_results, complex_results, comparison, output_dir):
    """Create human-readable calibration report"""

    report_lines = []
    report_lines.append("TRAFFIC MODEL CALIBRATION EVALUATION REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Tolerance Delta: {baseline_results['tolerance_delta']}")
    report_lines.append("")

    # Model summary
    report_lines.append("MODELS EVALUATED:")
    report_lines.append(f"  1. {baseline_results['model_name']}")
    report_lines.append(f"  2. {complex_results['model_name']}")
    report_lines.append("")

    # Macro averages comparison
    report_lines.append("MACRO-AVERAGED CALIBRATION METRICS:")
    report_lines.append("-" * 40)

    b_macro = baseline_results['macro_averages']
    c_macro = complex_results['macro_averages']
    macro_comp = comparison['macro_comparison']

    report_lines.append(f"{'Metric':<12} {'Baseline':<12} {'Complex':<12} {'Improvement':<12}")
    report_lines.append("-" * 48)
    report_lines.append(
        f"{'OPR':<12} {b_macro['OPR_macro']:<12.4f} {c_macro['OPR_macro']:<12.4f} {macro_comp['OPR']['improvement']:+12.4f}")
    report_lines.append(
        f"{'UPR':<12} {b_macro['UPR_macro']:<12.4f} {c_macro['UPR_macro']:<12.4f} {macro_comp['UPR']['improvement']:+12.4f}")
    report_lines.append(
        f"{'CR':<12} {b_macro['CR_macro']:<12.4f} {c_macro['CR_macro']:<12.4f} {macro_comp['CR']['improvement']:+12.4f}")
    report_lines.append(
        f"{'|SB|':<12} {macro_comp['SB_abs']['baseline']:<12.4f} {macro_comp['SB_abs']['complex']:<12.4f} {macro_comp['SB_abs']['improvement']:+12.4f}")
    report_lines.append("")

    # Micro averages comparison
    report_lines.append("MICRO-AVERAGED CALIBRATION METRICS:")
    report_lines.append("-" * 40)

    b_micro = baseline_results['micro_averages']
    c_micro = complex_results['micro_averages']
    micro_comp = comparison['micro_comparison']

    report_lines.append(f"{'Metric':<12} {'Baseline':<12} {'Complex':<12} {'Improvement':<12}")
    report_lines.append("-" * 48)
    report_lines.append(
        f"{'OPR':<12} {b_micro['OPR_micro']:<12.4f} {c_micro['OPR_micro']:<12.4f} {micro_comp['OPR']['improvement']:+12.4f}")
    report_lines.append(
        f"{'UPR':<12} {b_micro['UPR_micro']:<12.4f} {c_micro['UPR_micro']:<12.4f} {micro_comp['UPR']['improvement']:+12.4f}")
    report_lines.append(
        f"{'CR':<12} {b_micro['CR_micro']:<12.4f} {c_micro['CR_micro']:<12.4f} {micro_comp['CR']['improvement']:+12.4f}")
    report_lines.append(
        f"{'|SB|':<12} {micro_comp['SB_abs']['baseline']:<12.4f} {micro_comp['SB_abs']['complex']:<12.4f} {micro_comp['SB_abs']['improvement']:+12.4f}")
    report_lines.append("")

    # Overall assessment
    report_lines.append("OVERALL ASSESSMENT:")
    report_lines.append("-" * 20)
    assessment = comparison['overall_assessment']
    if assessment == "SIGNIFICANT_IMPROVEMENT":
        report_lines.append("✅ Complex model shows SIGNIFICANT calibration improvement")
    elif assessment == "MODERATE_IMPROVEMENT":
        report_lines.append("✅ Complex model shows MODERATE calibration improvement")
    elif assessment == "MARGINAL_IMPROVEMENT":
        report_lines.append("⚠️  Complex model shows MARGINAL calibration improvement")
    else:
        report_lines.append("❌ Complex model shows NO calibration improvement")

    report_lines.append("")
    report_lines.append("INTERPRETATION:")
    report_lines.append("- OPR: Over-Prediction Rate (lower is better)")
    report_lines.append("- UPR: Under-Prediction Rate (lower is better)")
    report_lines.append("- CR: Calibration Rate (higher is better)")
    report_lines.append("- |SB|: Absolute Sign Balance (lower is better)")
    report_lines.append("- Improvement = Baseline - Complex (for OPR/UPR/|SB|)")
    report_lines.append("- Improvement = Complex - Baseline (for CR)")

    # Save report with UTF-8 encoding to handle Unicode characters
    with open(os.path.join(output_dir, 'calibration_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    # Print to console
    print("\n" + '\n'.join(report_lines))


def create_calibration_visualizations(baseline_results, complex_results, output_dir):
    """Create calibration visualization plots"""

    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Calibration Comparison', fontsize=16, fontweight='bold')

        # Extract horizon-wise metrics
        baseline_horizons = [h['horizon'] for h in baseline_results['horizon_metrics']]
        complex_horizons = [h['horizon'] for h in complex_results['horizon_metrics']]

        baseline_opr = [h['OPR'] for h in baseline_results['horizon_metrics']]
        baseline_upr = [h['UPR'] for h in baseline_results['horizon_metrics']]
        baseline_cr = [h['CR'] for h in baseline_results['horizon_metrics']]
        baseline_sb = [h['SB'] for h in baseline_results['horizon_metrics']]

        complex_opr = [h['OPR'] for h in complex_results['horizon_metrics']]
        complex_upr = [h['UPR'] for h in complex_results['horizon_metrics']]
        complex_cr = [h['CR'] for h in complex_results['horizon_metrics']]
        complex_sb = [h['SB'] for h in complex_results['horizon_metrics']]

        # Plot 1: Over-Prediction Rate by Horizon
        axes[0, 0].plot(baseline_horizons, baseline_opr, 'o-', label='Baseline', linewidth=2, markersize=6)
        axes[0, 0].plot(complex_horizons, complex_opr, 's-', label='Complex', linewidth=2, markersize=6)
        axes[0, 0].set_title('Over-Prediction Rate (OPR) by Horizon', fontweight='bold')
        axes[0, 0].set_xlabel('Prediction Horizon')
        axes[0, 0].set_ylabel('OPR')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Under-Prediction Rate by Horizon
        axes[0, 1].plot(baseline_horizons, baseline_upr, 'o-', label='Baseline', linewidth=2, markersize=6)
        axes[0, 1].plot(complex_horizons, complex_upr, 's-', label='Complex', linewidth=2, markersize=6)
        axes[0, 1].set_title('Under-Prediction Rate (UPR) by Horizon', fontweight='bold')
        axes[0, 1].set_xlabel('Prediction Horizon')
        axes[0, 1].set_ylabel('UPR')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Calibration Rate by Horizon
        axes[1, 0].plot(baseline_horizons, baseline_cr, 'o-', label='Baseline', linewidth=2, markersize=6)
        axes[1, 0].plot(complex_horizons, complex_cr, 's-', label='Complex', linewidth=2, markersize=6)
        axes[1, 0].set_title('Calibration Rate (CR) by Horizon', fontweight='bold')
        axes[1, 0].set_xlabel('Prediction Horizon')
        axes[1, 0].set_ylabel('CR')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Sign Balance by Horizon
        axes[1, 1].plot(baseline_horizons, baseline_sb, 'o-', label='Baseline', linewidth=2, markersize=6)
        axes[1, 1].plot(complex_horizons, complex_sb, 's-', label='Complex', linewidth=2, markersize=6)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Sign Balance (SB) by Horizon', fontweight='bold')
        axes[1, 1].set_xlabel('Prediction Horizon')
        axes[1, 1].set_ylabel('SB')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'calibration_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Create summary bar chart
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        metrics = ['OPR', 'UPR', 'CR', '|SB|']
        baseline_values = [
            baseline_results['macro_averages']['OPR_macro'],
            baseline_results['macro_averages']['UPR_macro'],
            baseline_results['macro_averages']['CR_macro'],
            abs(baseline_results['macro_averages']['SB_macro'])
        ]
        complex_values = [
            complex_results['macro_averages']['OPR_macro'],
            complex_results['macro_averages']['UPR_macro'],
            complex_results['macro_averages']['CR_macro'],
            abs(complex_results['macro_averages']['SB_macro'])
        ]

        x = np.arange(len(metrics))
        width = 0.35

        ax.bar(x - width / 2, baseline_values, width, label='Baseline', alpha=0.8)
        ax.bar(x + width / 2, complex_values, width, label='Complex', alpha=0.8)

        ax.set_xlabel('Calibration Metrics', fontweight='bold')
        ax.set_ylabel('Metric Value', fontweight='bold')
        ax.set_title('Macro-Averaged Calibration Metrics Comparison', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (b_val, c_val) in enumerate(zip(baseline_values, complex_values)):
            ax.text(i - width / 2, b_val + 0.01, f'{b_val:.3f}', ha='center', va='bottom', fontsize=10)
            ax.text(i + width / 2, c_val + 0.01, f'{c_val:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'calibration_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print("Calibration visualizations saved successfully!")

    except Exception as e:
        print(f"Error creating visualizations: {e}")


if __name__ == "__main__":
    # Example usage with your file paths

    baseline_path = r"C:\Users\Fan\PycharmProjects\Msc_Thesis\baseline_model\baseline_outputs\baseline_simple_cnn_lstm_20250831_162807\models\best_baseline.pth"
    complex_path = r"C:\Users\Fan\PycharmProjects\Msc_Thesis\proposed\outputs\complex_center_1600_20250902_141604\models\best_model.pth"
    data_path = r"C:\Users\Fan\PycharmProjects\Msc_Thesis\processed_data\preprocessed_milan_traffic_center_1600_7day_splits_optimized.pkl"

    # Run comprehensive calibration evaluation
    baseline_results, complex_results, comparison = run_calibration_comparison(
        baseline_path=baseline_path,
        complex_path=complex_path,
        data_path=data_path,
        tolerance_delta=0.01  # 1% tolerance for [0,1] scaled data
    )

    print("Calibration evaluation completed successfully!")