#!/usr/bin/env python3
"""
Complete Complex Model with Unified Evaluation Pipeline
Addresses overfitting and evaluation consistency issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
import json
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr


class MultiHeadSpatialAttention(nn.Module):
    """Multi-head attention for spatial relationships"""

    def __init__(self, d_model, num_heads=4, dropout=0.1):  # Reduced from 8 heads
        super(MultiHeadSpatialAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size, seq_len, n_features = x.shape

        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.w_out(attended)
        return self.layer_norm(output + x)


class SimplifiedPeakBooster(nn.Module):
    """Simplified peak enhancement - reduced complexity"""

    def __init__(self, input_dim, n_squares):
        super(SimplifiedPeakBooster, self).__init__()

        # Single peak detector instead of multi-scale
        self.peak_detector = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )

        # Simplified boost factor
        self.boost_factor = nn.Parameter(torch.tensor(1.5))

    def forward(self, encoded_features, base_predictions):
        batch_size = encoded_features.shape[0]

        if len(encoded_features.shape) > 2:
            encoded_features = encoded_features.view(batch_size, -1)

        # Single peak detection
        peak_prob = self.peak_detector(encoded_features)

        # Apply boosting
        final_boost = 1.0 + (self.boost_factor - 1.0) * peak_prob
        boosted_predictions = base_predictions * final_boost

        return boosted_predictions, peak_prob


class ReasonableLoss(nn.Module):
    """Simplified loss function - more reasonable than your current 4-component loss"""

    def __init__(self, config=None):
        super(ReasonableLoss, self).__init__()

        # Simplified weights
        self.mse_weight = 0.6
        self.mae_weight = 0.4

        # Moderate peak enhancement
        self.peak_weight = 2.0  # Much lower than your 15.0

        self.mse_loss = nn.MSELoss(reduction='none')
        self.mae_loss = nn.L1Loss(reduction='none')

    def forward(self, predictions, targets, days_of_week=None):
        # Align shapes
        if predictions.shape != targets.shape:
            min_horizon = min(predictions.shape[1], targets.shape[1])
            min_squares = min(predictions.shape[2], targets.shape[2])
            predictions = predictions[:, :min_horizon, :min_squares]
            targets = targets[:, :min_horizon, :min_squares]

        # Base losses
        mse = self.mse_loss(predictions, targets)
        mae = self.mae_loss(predictions, targets)

        # Moderate peak emphasis
        target_flat = targets.view(-1)
        peak_threshold = torch.quantile(target_flat, 0.8)
        peak_mask = (targets > peak_threshold).float()

        # Apply moderate peak weighting
        weighted_mse = mse * (1.0 + peak_mask * (self.peak_weight - 1.0))

        total_loss = self.mse_weight * weighted_mse.mean() + self.mae_weight * mae.mean()

        return total_loss


class SimplifiedMultiScaleCNN(nn.Module):
    """Simplified multi-scale CNN - reduced from 7 to 3 kernels"""

    def __init__(self, input_dim, target_output_dim, kernel_sizes=[3, 5, 7]):  # Reduced complexity
        super(SimplifiedMultiScaleCNN, self).__init__()

        self.n_convs = len(kernel_sizes)
        base_dim = target_output_dim // self.n_convs
        remainder = target_output_dim % self.n_convs

        self.convs = nn.ModuleList()
        self.conv_dims = []

        for i, k in enumerate(kernel_sizes):
            current_dim = base_dim + (1 if i < remainder else 0)
            padding = k // 2

            conv_block = nn.Sequential(
                nn.Conv1d(input_dim, current_dim, kernel_size=k, padding=padding),
                nn.BatchNorm1d(current_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )

            self.convs.append(conv_block)
            self.conv_dims.append(current_dim)

        self.actual_output_dim = sum(self.conv_dims)
        print(f"SimplifiedMultiScaleCNN: kernels={kernel_sizes}, output_dim={self.actual_output_dim}")

    def forward(self, x):
        x = x.transpose(1, 2)

        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(x)
            conv_outputs.append(conv_out)

        output = torch.cat(conv_outputs, dim=1)
        return output.transpose(1, 2)


class ReasonableComplexModel(nn.Module):
    """
    Reasonably complex model - significant improvements over baseline without extreme overparameterization
    """

    def __init__(self, n_squares=1600, n_temporal_features=11, config=None):
        super(ReasonableComplexModel, self).__init__()

        # Reasonable complexity parameters
        self.n_squares = n_squares
        self.n_temporal_features = n_temporal_features
        self.hidden_dim = 256  # Reasonable size, not 768
        self.num_layers = 2  # Reasonable, not 4
        self.pred_horizon = 12

        print(f"Initializing Reasonable Complex Model:")
        print(f"  n_squares: {n_squares}")
        print(f"  hidden_dim: {self.hidden_dim}")
        print(f"  num_layers: {self.num_layers}")

        # Simplified multi-scale CNN
        self.traffic_cnn = SimplifiedMultiScaleCNN(n_squares, self.hidden_dim, [3, 5, 7])
        lstm_input_size = self.traffic_cnn.actual_output_dim

        # Reasonable LSTM layers
        self.lstm_layers = nn.ModuleList()
        current_input_size = lstm_input_size

        for i in range(self.num_layers):
            layer = nn.LSTM(
                input_size=current_input_size,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True,
                dropout=0.2 if i < self.num_layers - 1 else 0,
                bidirectional=True
            )
            self.lstm_layers.append(layer)
            current_input_size = self.hidden_dim * 2

        # Reasonable attention (4 heads instead of 8)
        self.spatial_attention = MultiHeadSpatialAttention(
            d_model=self.hidden_dim * 2,
            num_heads=4,
            dropout=0.1
        )

        # Simplified temporal processing
        self.temporal_encoder = nn.Sequential(
            nn.Linear(n_temporal_features, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Day embedding (reasonable size)
        self.day_embedding = nn.Embedding(7, self.hidden_dim // 4)

        # Simplified fusion
        fusion_input_dim = self.hidden_dim * 2 + self.hidden_dim + self.hidden_dim // 4

        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Single prediction head (not one per horizon)
        self.prediction_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, n_squares * self.pred_horizon)
        )

        # Single peak booster
        self.peak_booster = SimplifiedPeakBooster(self.hidden_dim, n_squares)

        print("Reasonable Complex Model initialized!")

    def forward(self, x_traffic, x_temporal):
        batch_size, seq_len, _ = x_traffic.shape

        # Extract day information
        day_probs = x_temporal[:, -1, :7]
        day_of_week = torch.argmax(day_probs, dim=1)

        # Multi-scale CNN
        x_traffic_processed = self.traffic_cnn(x_traffic)

        # LSTM processing
        lstm_out = x_traffic_processed
        for lstm_layer in self.lstm_layers:
            lstm_out, _ = lstm_layer(lstm_out)

        # Attention
        lstm_out = self.spatial_attention(lstm_out)
        final_hidden = lstm_out[:, -1, :]

        # Temporal features
        temporal_encoded = self.temporal_encoder(x_temporal[:, -1, :])

        # Day embedding
        day_embeds = self.day_embedding(day_of_week)

        # Fusion
        combined = torch.cat([final_hidden, temporal_encoded, day_embeds], dim=1)
        fused = self.fusion_network(combined)

        # Base predictions
        base_output = self.prediction_head(fused)
        base_output = base_output.view(batch_size, self.pred_horizon, self.n_squares)
        base_output = F.relu(base_output)

        # Peak boosting
        final_output, peak_prob = self.peak_booster(fused, base_output.mean(dim=1))

        # Apply peak boosting across all horizons
        final_output = final_output.unsqueeze(1).expand(-1, self.pred_horizon, -1)

        return final_output, day_of_week


class YourOriginalComplexModel(nn.Module):
    """
    Your original complex model - keeping all the complexity you had
    BUT with unified evaluation compatibility
    """

    def __init__(self, n_squares=1600, n_temporal_features=11, config=None):
        super(YourOriginalComplexModel, self).__init__()

        self.config = config
        self.n_squares = n_squares
        self.n_temporal_features = n_temporal_features
        self.hidden_dim = config.HIDDEN_DIM if config else 512  # Reduced from 768
        self.num_layers = config.NUM_LAYERS if config else 3  # Reduced from 4
        self.pred_horizon = config.PREDICTION_HORIZON if config else 12

        print(f"Initializing Your Original Complex Model:")
        print(f"  n_squares: {n_squares}")
        print(f"  hidden_dim: {self.hidden_dim}")
        print(f"  num_layers: {self.num_layers}")

        # Your complex multi-scale CNN (keeping all 7 kernels as you had)
        kernel_sizes = [1, 3, 5, 7, 11, 15, 21]
        self.traffic_cnn = ComplexMultiScaleCNN(n_squares, self.hidden_dim, kernel_sizes)
        lstm_input_size = self.traffic_cnn.actual_output_dim

        # Your enhanced LSTM layers
        self.lstm_layers = nn.ModuleList()
        current_input_size = lstm_input_size

        for i in range(self.num_layers):
            layer = nn.LSTM(
                input_size=current_input_size,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True,
                dropout=0.2 if i < self.num_layers - 1 else 0,
                bidirectional=True
            )
            self.lstm_layers.append(layer)
            current_input_size = self.hidden_dim * 2

        # Your spatial attention (reduced heads for stability)
        attention_heads = 4 if config and hasattr(config, 'ATTENTION_HEADS') else 4
        self.spatial_attention = MultiHeadSpatialAttention(
            d_model=self.hidden_dim * 2,
            num_heads=attention_heads,
            dropout=0.1
        )

        # Your temporal encoder
        self.temporal_encoder = nn.Sequential(
            nn.Linear(n_temporal_features, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Your embeddings
        self.day_embedding = nn.Embedding(7, self.hidden_dim // 2)
        self.hour_embedding = nn.Embedding(24, self.hidden_dim // 4)

        # Your fusion network
        fusion_input_dim = (self.hidden_dim * 2 + self.hidden_dim +
                            self.hidden_dim // 2 + self.hidden_dim // 4)

        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Simplified output (single head instead of multiple)
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, n_squares * self.pred_horizon)
        )

        # Single peak booster
        self.peak_booster = SimplifiedPeakBooster(self.hidden_dim, n_squares)

        # Learnable parameters (simplified)
        self.day_adjustment_weights = nn.Parameter(torch.ones(7))

        print("Your Original Complex Model initialized!")

    def forward(self, x_traffic, x_temporal):
        batch_size, seq_len, _ = x_traffic.shape

        # Extract temporal information
        day_probs = x_temporal[:, -1, :7]
        day_of_week = torch.argmax(day_probs, dim=1)
        hour_of_day = x_temporal[:, -1, 7].long()

        # Multi-scale CNN
        x_traffic_processed = self.traffic_cnn(x_traffic)

        # Multi-layer LSTM
        lstm_out = x_traffic_processed
        for lstm_layer in self.lstm_layers:
            lstm_out, _ = lstm_layer(lstm_out)

        # Spatial attention
        lstm_out = self.spatial_attention(lstm_out)
        final_hidden = lstm_out[:, -1, :]

        # Temporal features
        temporal_encoded = self.temporal_encoder(x_temporal[:, -1, :])

        # Embeddings
        day_embeds = self.day_embedding(day_of_week)
        hour_embeds = self.hour_embedding(hour_of_day)

        # Fusion
        combined = torch.cat([final_hidden, temporal_encoded, day_embeds, hour_embeds], dim=1)
        fused = self.fusion_network(combined)

        # Base predictions
        base_output = self.output_layer(fused)
        base_output = base_output.view(batch_size, self.pred_horizon, self.n_squares)
        base_output = F.relu(base_output)

        # Peak boosting (simplified)
        avg_output = base_output.mean(dim=1)  # Average across horizons for boosting
        boosted_avg, peak_prob = self.peak_booster(fused, avg_output)

        # Apply boost factor to all horizons
        boost_factor = boosted_avg / (avg_output + 1e-8)
        final_output = base_output * boost_factor.unsqueeze(1)

        # Day adjustments
        day_adjustment = self.day_adjustment_weights[day_of_week].unsqueeze(-1).unsqueeze(-1)
        final_output = final_output * day_adjustment

        return final_output, day_of_week


class ComplexMultiScaleCNN(nn.Module):
    """Your original complex CNN implementation"""

    def __init__(self, input_dim, target_output_dim, kernel_sizes=[1, 3, 5, 7, 11, 15, 21]):
        super(ComplexMultiScaleCNN, self).__init__()

        self.n_convs = len(kernel_sizes)
        base_dim = target_output_dim // self.n_convs
        remainder = target_output_dim % self.n_convs

        self.convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.conv_dims = []

        for i, k in enumerate(kernel_sizes):
            current_dim = base_dim + (1 if i < remainder else 0)
            padding = k // 2

            conv_block = nn.Sequential(
                nn.Conv1d(input_dim, current_dim, kernel_size=k, padding=padding),
                nn.BatchNorm1d(current_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv1d(current_dim, current_dim, kernel_size=k, padding=padding),
                nn.BatchNorm1d(current_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )

            residual_conv = nn.Conv1d(input_dim, current_dim, kernel_size=1)

            self.convs.append(conv_block)
            self.residual_convs.append(residual_conv)
            self.conv_dims.append(current_dim)

        self.actual_output_dim = sum(self.conv_dims)
        print(f"ComplexMultiScaleCNN: kernels={kernel_sizes}, output_dim={self.actual_output_dim}")

    def forward(self, x):
        x = x.transpose(1, 2)

        conv_outputs = []
        for conv, residual_conv in zip(self.convs, self.residual_convs):
            conv_out = conv(x)
            residual_out = residual_conv(x)
            combined = conv_out + residual_out
            conv_outputs.append(combined)

        output = torch.cat(conv_outputs, dim=1)
        return output.transpose(1, 2)


class UnifiedDataset(Dataset):
    """
    Unified dataset that works identically for baseline and complex models
    NO sophisticated augmentation to ensure fair comparison
    """

    def __init__(self, X_traffic, X_temporal, y, augment=False, config=None):
        self.X_traffic = torch.FloatTensor(X_traffic)
        self.X_temporal = torch.FloatTensor(X_temporal)
        self.y = torch.FloatTensor(y)
        self.augment = augment  # Keep for compatibility but use simple augmentation

    def __len__(self):
        return len(self.X_traffic)

    def __getitem__(self, idx):
        x_traffic = self.X_traffic[idx].clone()
        x_temporal = self.X_temporal[idx].clone()
        y = self.y[idx].clone()

        # Simple augmentation (same as baseline) for fair comparison
        if self.augment and torch.rand(1).item() < 0.1:
            noise_scale = 0.01
            traffic_noise = torch.randn_like(x_traffic) * noise_scale
            x_traffic = torch.clamp(x_traffic + traffic_noise, min=0, max=1)

        return x_traffic, x_temporal, y


class UnifiedEvaluator:
    """
    Unified evaluator that ensures identical evaluation for both models
    """

    def __init__(self, device='cpu'):
        self.device = device

    def evaluate_model(self, model, X_traffic, X_temporal, y_true, batch_size=16):
        """Evaluate model with identical methodology"""

        model.eval()
        model = model.to(self.device)

        # Use unified dataset (no augmentation)
        dataset = UnifiedDataset(X_traffic, X_temporal, y_true, augment=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        all_predictions = []
        all_actuals = []
        all_days = []

        with torch.no_grad():
            for x_traffic, x_temporal, y in dataloader:
                x_traffic = x_traffic.to(self.device)
                x_temporal = x_temporal.to(self.device)

                # Handle different model outputs uniformly
                output = model(x_traffic, x_temporal)
                if isinstance(output, tuple):
                    predictions, day_of_week = output
                else:
                    predictions = output
                    day_of_week = torch.argmax(x_temporal[:, -1, :7], dim=1)

                all_predictions.append(predictions.cpu().numpy())
                all_actuals.append(y.numpy())
                all_days.append(day_of_week.cpu().numpy())

        predictions_array = np.concatenate(all_predictions, axis=0)
        actuals_array = np.concatenate(all_actuals, axis=0)
        days_array = np.concatenate(all_days, axis=0)

        return predictions_array, actuals_array, days_array

    def calculate_metrics(self, predictions, actuals, days_of_week, split_name="unknown"):
        """Calculate metrics with identical methodology"""

        # IDENTICAL flattening for fair comparison
        pred_flat = predictions.reshape(-1)
        actual_flat = actuals.reshape(-1)

        # Basic metrics
        mae = float(mean_absolute_error(actual_flat, pred_flat))
        rmse = float(np.sqrt(mean_squared_error(actual_flat, pred_flat)))
        r2 = float(r2_score(actual_flat, pred_flat))

        # Fixed MAPE calculation
        epsilon = 1e-3  # Appropriate for normalized data
        mape = float(np.mean(np.abs((actual_flat - pred_flat) / (actual_flat + epsilon))) * 100)

        correlation, _ = pearsonr(actual_flat, pred_flat)
        correlation = float(correlation)

        overpred_rate = float(np.mean(pred_flat > actual_flat) * 100)

        # Day-specific metrics
        day_metrics = {}
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        for day in range(7):
            day_mask = (days_of_week == day)
            if np.sum(day_mask) > 0:
                day_pred = predictions[day_mask].reshape(-1)
                day_actual = actuals[day_mask].reshape(-1)

                day_metrics[day_names[day]] = {
                    'mae': float(mean_absolute_error(day_actual, day_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(day_actual, day_pred))),
                    'r2': float(r2_score(day_actual, day_pred)) if len(day_actual) > 1 else 0.0,
                    'samples': int(np.sum(day_mask))
                }

        return {
            'overall': {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'correlation': correlation,
                'overprediction_rate': overpred_rate,
                'samples': len(pred_flat)
            },
            'day_specific': day_metrics,
            'split_name': split_name
        }


def compare_with_baseline(complex_model, baseline_model, data, output_dir):
    """
    Fair comparison between your complex model and baseline
    """
    print("=" * 80)
    print("FAIR MODEL COMPARISON - UNIFIED EVALUATION")
    print("=" * 80)

    evaluator = UnifiedEvaluator()

    # Evaluate complex model
    print("Evaluating complex model...")
    complex_val_pred, complex_val_actual, complex_val_days = evaluator.evaluate_model(
        complex_model, data['X_traffic_val'], data['X_temporal_val'], data['y_val'], batch_size=8
    )
    complex_val_metrics = evaluator.calculate_metrics(
        complex_val_pred, complex_val_actual, complex_val_days, "validation"
    )

    complex_test_pred, complex_test_actual, complex_test_days = evaluator.evaluate_model(
        complex_model, data['X_traffic_test'], data['X_temporal_test'], data['y_test'], batch_size=8
    )
    complex_test_metrics = evaluator.calculate_metrics(
        complex_test_pred, complex_test_actual, complex_test_days, "test"
    )

    # Evaluate baseline model
    print("Evaluating baseline model...")
    baseline_val_pred, baseline_val_actual, baseline_val_days = evaluator.evaluate_model(
        baseline_model, data['X_traffic_val'], data['X_temporal_val'], data['y_val'], batch_size=16
    )
    baseline_val_metrics = evaluator.calculate_metrics(
        baseline_val_pred, baseline_val_actual, baseline_val_days, "validation"
    )

    baseline_test_pred, baseline_test_actual, baseline_test_days = evaluator.evaluate_model(
        baseline_model, data['X_traffic_test'], data['X_temporal_test'], data['y_test'], batch_size=16
    )
    baseline_test_metrics = evaluator.calculate_metrics(
        baseline_test_pred, baseline_test_actual, baseline_test_days, "test"
    )

    # Print comparison
    print("\n" + "=" * 100)
    print("UNIFIED COMPARISON RESULTS")
    print("=" * 100)

    print(f"{'Metric':<15} {'Baseline Val':<15} {'Complex Val':<15} {'Baseline Test':<15} {'Complex Test':<15}")
    print("-" * 75)

    bv = baseline_val_metrics['overall']
    cv = complex_val_metrics['overall']
    bt = baseline_test_metrics['overall']
    ct = complex_test_metrics['overall']

    print(f"{'MAE':<15} {bv['mae']:<15.6f} {cv['mae']:<15.6f} {bt['mae']:<15.6f} {ct['mae']:<15.6f}")
    print(f"{'RMSE':<15} {bv['rmse']:<15.6f} {cv['rmse']:<15.6f} {bt['rmse']:<15.6f} {ct['rmse']:<15.6f}")
    print(f"{'RÃ‚Â²':<15} {bv['r2']:<15.6f} {cv['r2']:<15.6f} {bt['r2']:<15.6f} {ct['r2']:<15.6f}")
    print(f"{'MAPE (%)':<15} {bv['mape']:<15.2f} {cv['mape']:<15.2f} {bt['mape']:<15.2f} {ct['mape']:<15.2f}")
    print(
        f"{'Correlation':<15} {bv['correlation']:<15.6f} {cv['correlation']:<15.6f} {bt['correlation']:<15.6f} {ct['correlation']:<15.6f}")

    # Improvement analysis
    print("\n" + "=" * 50)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 50)

    r2_improvement = ct['r2'] - bt['r2']
    mae_improvement = ((bt['mae'] - ct['mae']) / bt['mae'] * 100)

    print(f"Test RÃ‚Â² improvement: {r2_improvement:+.6f}")
    print(f"Test MAE improvement: {mae_improvement:+.2f}%")

    # Overfitting check
    baseline_gap = abs(bv['r2'] - bt['r2'])
    complex_gap = abs(cv['r2'] - ct['r2'])

    print(f"\nGeneralization Analysis:")
    print(f"  Baseline val-test gap: {baseline_gap:.6f}")
    print(f"  Complex val-test gap:  {complex_gap:.6f}")

    if complex_gap > baseline_gap + 0.02:
        print("  WARNING: Complex model shows overfitting!")
    elif r2_improvement > 0.01 and mae_improvement > 2:
        print("  SUCCESS: Complex model shows meaningful improvement")
    else:
        print("  MARGINAL: Complex model provides minimal improvement")

    # Save results
    comparison_results = {
        'baseline': {'validation': baseline_val_metrics, 'test': baseline_test_metrics},
        'complex': {'validation': complex_val_metrics, 'test': complex_test_metrics},
        'improvements': {
            'r2_improvement': r2_improvement,
            'mae_improvement_percent': mae_improvement,
            'baseline_generalization_gap': baseline_gap,
            'complex_generalization_gap': complex_gap
        }
    }

    with open(os.path.join(output_dir, 'unified_comparison.json'), 'w') as f:
        json.dump(comparison_results, f, indent=2)

    print(f"\nComparison results saved to: {output_dir}/unified_comparison.json")
    return comparison_results


def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def create_reasonable_complex_model(n_squares, n_temporal_features, use_original_complexity=False):
    """
    Create either a reasonable or your original complex model
    """

    class ReasonableConfig:
        HIDDEN_DIM = 256
        NUM_LAYERS = 2
        PREDICTION_HORIZON = 12
        ATTENTION_HEADS = 4

    class YourOriginalConfig:
        HIDDEN_DIM = 512  # Reduced from 768 for stability
        NUM_LAYERS = 3  # Reduced from 4
        PREDICTION_HORIZON = 12
        ATTENTION_HEADS = 4  # Reduced from 8

    config = YourOriginalConfig() if use_original_complexity else ReasonableConfig()

    if use_original_complexity:
        model = YourOriginalComplexModel(n_squares, n_temporal_features, config)
        model_type = "Your Original Complex"
    else:
        model = ReasonableComplexModel(n_squares, n_temporal_features, config)
        model_type = "Reasonable Complex"

    total_params, trainable_params = count_parameters(model)

    print(f"\n{model_type} Model Created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")

    return model, config


# Aliases for compatibility with your existing code
ComplexDayAwareLSTMTrafficPredictor = YourOriginalComplexModel
ComplexDayAwareMilanTrafficDataset = UnifiedDataset
ComplexDayOfWeekAwareLoss = ReasonableLoss
ComplexPeakLoss = ReasonableLoss


# Calibrator (simplified)
class ComplexDayAwareCalibrator:
    def __init__(self, config=None):
        self.day_factors = {i: 1.0 for i in range(7)}
        self.fitted = False

    def fit(self, predictions, actuals, days_of_week, hours=None):
        for day in range(7):
            day_mask = (days_of_week == day)
            if day_mask.sum() > 10:
                day_preds = predictions[day_mask]
                day_actuals = actuals[day_mask]
                if day_preds.mean() > 0:
                    factor = day_actuals.mean() / day_preds.mean()
                    self.day_factors[day] = np.clip(factor, 0.8, 1.2)
        self.fitted = True

    def calibrate(self, predictions, days_of_week):
        if not self.fitted:
            return predictions

        calibrated = predictions.copy()
        for day in range(7):
            day_mask = (days_of_week == day)
            if day_mask.sum() > 0:
                calibrated[day_mask] *= self.day_factors[day]
        return calibrated


def create_complex_model(config, n_squares, n_temporal_features):
    """Factory function for your existing code compatibility"""
    return create_reasonable_complex_model(n_squares, n_temporal_features, use_original_complexity=True)[0]


if __name__ == "__main__":
    print("Complete Complex Model Implementation")
    print("Includes unified evaluation for fair comparison with baseline")

    # Test model creation
    model, config = create_reasonable_complex_model(1600, 11, use_original_complexity=True)
    print(f"Model created successfully with {sum(p.numel() for p in model.parameters()):,} parameters")