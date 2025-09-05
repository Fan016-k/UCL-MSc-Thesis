#!/usr/bin/env python3
"""
Simple CNN-LSTM Baseline Model for Traffic Prediction
Designed for R² around 0.7 performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class TrafficLoss(nn.Module):
    """
    Simple but effective loss function for traffic prediction
    """

    def __init__(self):
        super(TrafficLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, predictions, targets):
        # Ensure shapes match
        if predictions.shape != targets.shape:
            min_horizon = min(predictions.shape[1], targets.shape[1])
            min_squares = min(predictions.shape[2], targets.shape[2])
            predictions = predictions[:, :min_horizon, :min_squares]
            targets = targets[:, :min_horizon, :min_squares]

        # Combine MSE and MAE losses
        mse_loss = self.mse(predictions, targets)
        mae_loss = self.mae(predictions, targets)

        # Weighted combination
        return 0.6 * mse_loss + 0.4 * mae_loss


class TrafficDataset:
    """
    Simple dataset wrapper for traffic data
    """

    def __init__(self, X_traffic, X_temporal, y, augment=False):
        self.X_traffic = torch.FloatTensor(X_traffic)
        self.X_temporal = torch.FloatTensor(X_temporal)
        self.y = torch.FloatTensor(y)
        self.augment = augment

    def __len__(self):
        return len(self.X_traffic)

    def __getitem__(self, idx):
        x_traffic = self.X_traffic[idx].clone()
        x_temporal = self.X_temporal[idx].clone()
        y = self.y[idx].clone()

        # Simple augmentation for training
        if self.augment and torch.rand(1).item() < 0.1:
            noise_scale = 0.01
            traffic_noise = torch.randn_like(x_traffic) * noise_scale
            x_traffic = torch.clamp(x_traffic + traffic_noise, min=0, max=1)

        return x_traffic, x_temporal, y


class BaselineTrainer:
    """
    Simple trainer for the baseline model
    """

    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config

        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Loss function
        self.criterion = TrafficLoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=config.PATIENCE // 3,
            factor=0.5
        )

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        valid_batches = 0

        for batch_idx, (x_traffic, x_temporal, y) in enumerate(train_loader):
            try:
                x_traffic = x_traffic.to(self.device)
                x_temporal = x_temporal.to(self.device)
                y = y.to(self.device)

                # Check for NaN inputs
                if torch.isnan(x_traffic).any() or torch.isnan(x_temporal).any() or torch.isnan(y).any():
                    continue

                self.optimizer.zero_grad()
                predictions, _ = self.model(x_traffic, x_temporal)

                loss = self.criterion(predictions, y)

                # Check for valid loss
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                valid_batches += 1

                if batch_idx % self.config.LOG_FREQUENCY == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        if valid_batches == 0:
            return float('inf')

        avg_loss = epoch_loss / valid_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        valid_batches = 0

        with torch.no_grad():
            for x_traffic, x_temporal, y in val_loader:
                try:
                    x_traffic = x_traffic.to(self.device)
                    x_temporal = x_temporal.to(self.device)
                    y = y.to(self.device)

                    predictions, _ = self.model(x_traffic, x_temporal)
                    loss = self.criterion(predictions, y)

                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        val_loss += loss.item()
                        valid_batches += 1

                except Exception as e:
                    continue

        if valid_batches == 0:
            return float('inf')

        avg_loss = val_loss / valid_batches
        self.val_losses.append(avg_loss)
        return avg_loss

    def should_stop_early(self, val_loss):
        """Check if training should stop early"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False, True  # continue training, is_best
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.PATIENCE, False

    def step_scheduler(self, val_loss):
        """Step the learning rate scheduler"""
        self.scheduler.step(val_loss)


def train_baseline_model(config):
    """
    Main function to train the baseline model
    """
    import pickle
    from torch.utils.data import DataLoader
    import os
    import json
    from datetime import datetime

    print("=" * 80)
    print("TRAINING SIMPLE CNN-LSTM BASELINE MODEL")
    print("Target Performance: R² ≈ 0.7")
    print("=" * 80)

    # Load data
    print(f"Loading data from: {config.DATA_PATH}")
    with open(config.DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    n_squares = data['X_traffic_train'].shape[2]
    n_temporal_features = data['X_temporal_train'].shape[2]

    print(f"Data loaded:")
    print(f"  n_squares: {n_squares}")
    print(f"  n_temporal_features: {n_temporal_features}")
    print(f"  Train samples: {data['X_traffic_train'].shape[0]}")
    print(f"  Val samples: {data['X_traffic_val'].shape[0]}")
    print(f"  Test samples: {data['X_traffic_test'].shape[0]}")

    # Create model
    model = SimpleBaselineLSTM(
        n_squares=n_squares,
        n_temporal_features=n_temporal_features,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        prediction_horizon=config.PREDICTION_HORIZON
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and config.USE_CUDA else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")

    # Create datasets
    train_dataset = TrafficDataset(
        data['X_traffic_train'],
        data['X_temporal_train'],
        data['y_train'],
        augment=True
    )
    val_dataset = TrafficDataset(
        data['X_traffic_val'],
        data['X_temporal_val'],
        data['y_val'],
        augment=False
    )
    test_dataset = TrafficDataset(
        data['X_traffic_test'],
        data['X_temporal_test'],
        data['y_test'],
        augment=False
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    # Create trainer
    trainer = BaselineTrainer(model, device, config)

    # Create output directory
    output_dir = config.create_output_folder()
    print(f"Output directory: {output_dir}")

    # Training loop
    print(f"Starting training for {config.NUM_EPOCHS} epochs...")

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")

        # Train
        train_loss = trainer.train_epoch(train_loader, epoch)

        # Validate
        val_loss = trainer.validate(val_loader)

        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Check for early stopping
        should_stop, is_best = trainer.should_stop_early(val_loss)

        if is_best:
            print(f"  New best model! Validation loss: {val_loss:.6f}")
            # Save best model
            # Create serializable config dict
            config_dict = {}
            for key in dir(config):
                if not key.startswith('_') and not callable(getattr(config, key)):
                    value = getattr(config, key)
                    if isinstance(value, (int, float, str, bool, list, dict)):
                        config_dict[key] = value
                    else:
                        config_dict[key] = str(value)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_loss': val_loss,
                'train_losses': trainer.train_losses,
                'val_losses': trainer.val_losses,
                'config': config_dict
            }, os.path.join(output_dir, 'models', 'best_baseline.pth'))
        else:
            print(f"  Patience: {trainer.patience_counter}/{config.PATIENCE}")

        # Step scheduler
        trainer.step_scheduler(val_loss)

        # Early stopping
        if should_stop:
            print(f"Early stopping after {epoch + 1} epochs")
            break

        # Save checkpoint
        if (epoch + 1) % config.CHECKPOINT_FREQUENCY == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_loss': val_loss,
                'train_losses': trainer.train_losses,
                'val_losses': trainer.val_losses
            }, os.path.join(output_dir, 'checkpoints', f'checkpoint_epoch_{epoch + 1}.pth'))

    # Test the final model
    print("\nTesting baseline model...")

    # Load best model
    checkpoint = torch.load(os.path.join(output_dir, 'models', 'best_baseline.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test evaluation
    model.eval()
    test_predictions = []
    test_actuals = []
    test_days = []

    with torch.no_grad():
        for x_traffic, x_temporal, y in test_loader:
            try:
                x_traffic = x_traffic.to(device)
                x_temporal = x_temporal.to(device)

                predictions, day_of_week = model(x_traffic, x_temporal)

                test_predictions.append(predictions.cpu().numpy())
                test_actuals.append(y.numpy())
                test_days.append(day_of_week.cpu().numpy())

            except Exception as e:
                print(f"Error in test batch: {e}")
                continue

    if test_predictions:
        test_predictions = np.concatenate(test_predictions, axis=0)
        test_actuals = np.concatenate(test_actuals, axis=0)
        test_days = np.concatenate(test_days, axis=0)

        # Calculate metrics
        pred_flat = test_predictions.reshape(-1)
        actual_flat = test_actuals.reshape(-1)

        mae = np.mean(np.abs(pred_flat - actual_flat))
        rmse = np.sqrt(np.mean((pred_flat - actual_flat) ** 2))

        # R² calculation
        ss_res = np.sum((actual_flat - pred_flat) ** 2)
        ss_tot = np.sum((actual_flat - np.mean(actual_flat)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        overpred_rate = np.mean(pred_flat > actual_flat) * 100

        # Non-zero traffic performance
        non_zero_mask = actual_flat > 0
        if np.sum(non_zero_mask) > 0:
            mae_nonzero = np.mean(np.abs(pred_flat[non_zero_mask] - actual_flat[non_zero_mask]))
            r2_nonzero_ss_res = np.sum((actual_flat[non_zero_mask] - pred_flat[non_zero_mask]) ** 2)
            r2_nonzero_ss_tot = np.sum((actual_flat[non_zero_mask] - np.mean(actual_flat[non_zero_mask])) ** 2)
            r2_nonzero = 1 - (r2_nonzero_ss_res / r2_nonzero_ss_tot)
        else:
            mae_nonzero = 0
            r2_nonzero = 0

        print(f"\n" + "=" * 80)
        print("BASELINE MODEL TEST RESULTS:")
        print("=" * 80)
        print(f"Overall Performance:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Overprediction Rate: {overpred_rate:.1f}%")
        print(f"  Best Validation Loss: {trainer.best_val_loss:.6f}")
        print(f"")
        print(f"Non-zero Traffic Performance:")
        print(f"  MAE (non-zero): {mae_nonzero:.4f}")
        print(f"  R² (non-zero): {r2_nonzero:.4f}")
        print(
            f"  Non-zero samples: {np.sum(non_zero_mask):,}/{len(actual_flat):,} ({np.sum(non_zero_mask) / len(actual_flat) * 100:.1f}%)")

        # Performance assessment
        if r2 >= 0.7:
            print(f"\n✅ Excellent! R² ≥ 0.7 - Target performance achieved!")
        elif r2 >= 0.5:
            print(f"\n✅ Good! R² ≥ 0.5 - Strong baseline performance")
        elif r2 >= 0.3:
            print(f"\n⚠️  Moderate performance - R² ≥ 0.3")
        elif r2 > 0:
            print(f"\n⚠️  Low but positive R² - baseline needs improvement")
        else:
            print(f"\n❌ Negative R² - model worse than mean prediction")

        # Save results with proper JSON serialization
        def make_serializable(obj):
            """Convert any object to JSON-serializable format"""
            if isinstance(obj, dict):
                return {key: make_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                return str(obj)  # Convert complex objects to string
            else:
                try:
                    json.dumps(obj)  # Test if serializable
                    return obj
                except:
                    return str(obj)  # Convert to string if not serializable

        # Create JSON-serializable config
        config_dict = {}
        for key in dir(config):
            if not key.startswith('_') and not callable(getattr(config, key)):
                value = getattr(config, key)
                config_dict[key] = make_serializable(value)

        results = {
            'test_mae': float(mae),
            'test_rmse': float(rmse),
            'test_r2': float(r2),
            'test_overprediction_rate': float(overpred_rate),
            'test_mae_nonzero': float(mae_nonzero),
            'test_r2_nonzero': float(r2_nonzero),
            'best_val_loss': float(trainer.best_val_loss),
            'total_parameters': int(total_params),
            'epochs_trained': len(trainer.train_losses),
            'model_path': os.path.join(output_dir, 'models', 'best_baseline.pth'),
            'config_used': config_dict
        }

        # Ensure results are fully serializable
        results = make_serializable(results)

        with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        # Save config
        config.save_config(output_dir)

        print(f"\nModel and results saved to: {output_dir}")
        print("=" * 80)

        return results, output_dir

    else:
        print("❌ No test predictions generated!")
        return None, output_dir


def load_trained_model(model_path, n_squares=1600, n_temporal_features=11):
    """
    Load a trained baseline model
    """
    print(f"Loading trained model from: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu')

    # Get model parameters from checkpoint if available
    config_dict = checkpoint.get('config', {})
    hidden_dim = config_dict.get('HIDDEN_DIM', 256)
    num_layers = config_dict.get('NUM_LAYERS', 2)
    prediction_horizon = config_dict.get('PREDICTION_HORIZON', 12)

    # Create model
    model = SimpleBaselineLSTM(
        n_squares=n_squares,
        n_temporal_features=n_temporal_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        prediction_horizon=prediction_horizon
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Model loaded successfully!")
    print(f"Best validation loss: {checkpoint.get('val_loss', 'N/A')}")

    return model, checkpoint


def evaluate_model(model, data, device='cpu'):
    """
    Evaluate model performance on validation and test sets
    """
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    print("Evaluating model performance...")

    model = model.to(device)
    model.eval()

    # Create datasets
    val_dataset = TrafficDataset(data['X_traffic_val'], data['X_temporal_val'], data['y_val'])
    test_dataset = TrafficDataset(data['X_traffic_test'], data['X_temporal_test'], data['y_test'])

    # Create data loaders
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    def generate_predictions(data_loader):
        predictions = []
        actuals = []
        days = []

        with torch.no_grad():
            for x_traffic, x_temporal, y in data_loader:
                x_traffic, x_temporal = x_traffic.to(device), x_temporal.to(device)
                pred, day_of_week = model(x_traffic, x_temporal)

                predictions.append(pred.cpu().numpy())
                actuals.append(y.numpy())
                days.append(day_of_week.cpu().numpy())

        return (np.concatenate(predictions, axis=0),
                np.concatenate(actuals, axis=0),
                np.concatenate(days, axis=0))

    # Generate predictions
    val_predictions, val_actuals, val_days = generate_predictions(val_loader)
    test_predictions, test_actuals, test_days = generate_predictions(test_loader)

    # Calculate comprehensive metrics
    results = {}

    for split_name, (predictions, actuals) in [('validation', (val_predictions, val_actuals)),
                                               ('test', (test_predictions, test_actuals))]:
        pred_flat = predictions.reshape(-1)
        actual_flat = actuals.reshape(-1)

        results[split_name] = {
            'mae': float(mean_absolute_error(actual_flat, pred_flat)),
            'rmse': float(np.sqrt(mean_squared_error(actual_flat, pred_flat))),
            'r2': float(r2_score(actual_flat, pred_flat)),
            'overprediction_rate': float(np.mean(pred_flat > actual_flat) * 100),
            'samples': len(pred_flat)
        }

    return results, (val_predictions, val_actuals, val_days), (test_predictions, test_actuals, test_days)