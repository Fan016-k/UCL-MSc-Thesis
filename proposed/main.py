#!/usr/bin/env python3
"""
Windows-Safe Training script for Complex Day-Aware Traffic Prediction Model - CENTER 1600 SQUARES
Updated for complex architecture with attention mechanisms and multi-scale processing
"""
import os

# Windows-specific fixes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
import time
import logging
from tqdm import tqdm
import warnings
import gc  # Garbage collection

warnings.filterwarnings('ignore')

# Import your modules - Updated for complex architecture
from config import  WindowsConfig
from model import (
    ComplexDayAwareLSTMTrafficPredictor,
    ComplexDayAwareMilanTrafficDataset,
    ComplexDayOfWeekAwareLoss,
    ComplexDayAwareCalibrator,
    count_parameters,
    create_complex_model
)
from visualization import (
    analyze_day_specific_performance,
    plot_training_history
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ComplexWindowsSafeTrainer:
    """Windows-safe trainer class for complex architecture traffic prediction"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.USE_CUDA else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Display model complexity info
        complexity_info = config.get_model_complexity_info()
        logger.info(f"Complex Model Info: {complexity_info['architecture_type']}")
        logger.info(f"  Hidden dim: {complexity_info['hidden_dim']}")
        logger.info(f"  Attention heads: {complexity_info['attention_heads']}")
        logger.info(f"  Estimated parameters: {complexity_info['estimated_parameters']}")

        # Force single-threaded operation for Windows stability
        torch.set_num_threads(1)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.day_specific_losses = {day: [] for day in range(7)}

        # Best model tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def load_data(self):
        """Load preprocessed center region data with memory optimization"""
        logger.info(f"Loading CENTER 1600 SQUARES data from: {self.config.DATA_PATH}")

        try:
            with open(self.config.DATA_PATH, 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.config.DATA_PATH}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

        # Extract dimensions
        self.n_squares = data['X_traffic_train'].shape[2]
        self.n_temporal_features = data['X_temporal_train'].shape[2]

        logger.info(f"COMPLEX MODEL Data loaded successfully!")
        logger.info(f"  n_squares: {self.n_squares}")
        logger.info(f"  n_temporal_features: {self.n_temporal_features}")
        logger.info(f"  Train samples: {len(data['X_traffic_train'])}")
        logger.info(f"  Val samples: {len(data['X_traffic_val'])}")
        logger.info(f"  Test samples: {len(data['X_traffic_test'])}")

        # Verify we have 1600 squares
        if self.n_squares != 1600:
            logger.warning(f"Expected 1600 squares, but got {self.n_squares}")
        else:
            logger.info("Confirmed: Working with center 1600 squares as expected!")

        return data

    def create_dataloaders(self, data):
        """Create data loaders with Windows-safe settings for complex model"""
        logger.info("Creating Windows-safe data loaders for COMPLEX MODEL...")

        # Create datasets using complex dataset class
        train_dataset = ComplexDayAwareMilanTrafficDataset(
            data['X_traffic_train'],
            data['X_temporal_train'],
            data['y_train'],
            augment=self.config.USE_ADVANCED_AUGMENTATION,
            config=self.config
        )

        val_dataset = ComplexDayAwareMilanTrafficDataset(
            data['X_traffic_val'],
            data['X_temporal_val'],
            data['y_val'],
            augment=False,
            config=self.config
        )

        test_dataset = ComplexDayAwareMilanTrafficDataset(
            data['X_traffic_test'],
            data['X_temporal_test'],
            data['y_test'],
            augment=False,
            config=self.config
        )

        logger.info(f"Complex Dataset sizes:")
        logger.info(f"  Train: {len(train_dataset)} samples")
        logger.info(f"  Validation: {len(val_dataset)} samples")
        logger.info(f"  Test: {len(test_dataset)} samples")

        # Windows-safe data loaders (no multiprocessing)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,  # No multiprocessing
            pin_memory=False,  # Disabled for Windows
            drop_last=True  # Avoid partial batches
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,  # No multiprocessing
            pin_memory=False,  # Disabled for Windows
            drop_last=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,  # No multiprocessing
            pin_memory=False  # Disabled for Windows
        )

        logger.info(f"Complex model data loaders created:")
        logger.info(f"  Batch size: {self.config.BATCH_SIZE}")
        logger.info(f"  Advanced augmentation: {self.config.USE_ADVANCED_AUGMENTATION}")

        return train_loader, val_loader, test_loader

    def create_model(self):
        """Create complex model with Windows-safe settings"""
        logger.info("Creating COMPLEX Windows-safe model...")

        # Use the complex model
        model = ComplexDayAwareLSTMTrafficPredictor(
            n_squares=self.n_squares,
            n_temporal_features=self.n_temporal_features,
            config=self.config
        )

        # Count parameters
        total_params, trainable_params = count_parameters(model)
        model_size_mb = (total_params * 4) / (1024 * 1024)

        logger.info(f"COMPLEX model created:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Estimated model size: {model_size_mb:.1f} MB")
        logger.info(f"  Hidden dimension: {self.config.HIDDEN_DIM}")
        logger.info(f"  Number of layers: {self.config.NUM_LAYERS}")
        logger.info(f"  Attention heads: {self.config.ATTENTION_HEADS}")

        return model.to(self.device)

    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Windows-safe training epoch for complex model"""
        model.train()
        epoch_loss = 0
        batch_count = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.config.NUM_EPOCHS} [Complex Train]')

        for batch_idx, (x_traffic, x_temporal, y) in enumerate(pbar):
            try:
                x_traffic = x_traffic.to(self.device)
                x_temporal = x_temporal.to(self.device)
                y = y.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                predictions, day_of_week = model(x_traffic, x_temporal)

                # Calculate complex loss
                if self.config.USE_COMPLEX_LOSS:
                    loss = criterion(predictions, y, day_of_week)
                else:
                    loss = nn.MSELoss()(predictions, y)

                # Backward pass
                loss.backward()

                # Gradient clipping for complex model
                if self.config.GRADIENT_CLIP_VALUE:
                    nn.utils.clip_grad_norm_(model.parameters(), self.config.GRADIENT_CLIP_VALUE)

                optimizer.step()

                # Update metrics
                epoch_loss += loss.item()
                batch_count += 1

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.8f}'
                })

                # Memory cleanup every 10 batches (more frequent for complex model)
                if batch_idx % 10 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue

        return epoch_loss / max(batch_count, 1)

    def validate(self, model, val_loader, criterion):
        """Windows-safe validation for complex model"""
        model.eval()
        val_loss = 0
        batch_count = 0
        day_losses = {day: [] for day in range(7)}

        with torch.no_grad():
            for x_traffic, x_temporal, y in val_loader:
                try:
                    x_traffic = x_traffic.to(self.device)
                    x_temporal = x_temporal.to(self.device)
                    y = y.to(self.device)

                    # Forward pass
                    predictions, day_of_week = model(x_traffic, x_temporal)

                    # Calculate complex loss
                    if self.config.USE_COMPLEX_LOSS:
                        loss = criterion(predictions, y, day_of_week)
                    else:
                        loss = nn.MSELoss()(predictions, y)

                    val_loss += loss.item()
                    batch_count += 1

                    # Day-specific losses
                    for day in range(7):
                        day_mask = (day_of_week == day)
                        if day_mask.any():
                            day_pred = predictions[day_mask]
                            day_actual = y[day_mask]
                            day_loss = nn.MSELoss()(day_pred, day_actual)
                            day_losses[day].append(day_loss.item())

                except Exception as e:
                    logger.warning(f"Error in validation batch: {e}")
                    continue

        # Average day-specific losses
        for day in range(7):
            if day_losses[day]:
                self.day_specific_losses[day].append(np.mean(day_losses[day]))

        return val_loss / max(batch_count, 1)

    def test_model(self, model, test_loader):
        """Windows-safe model testing for complex model"""
        model.eval()
        predictions = []
        actuals = []
        days = []

        with torch.no_grad():
            for x_traffic, x_temporal, y in test_loader:
                try:
                    x_traffic = x_traffic.to(self.device)
                    x_temporal = x_temporal.to(self.device)

                    output, day_of_week = model(x_traffic, x_temporal)

                    predictions.append(output.cpu().numpy())
                    actuals.append(y.numpy())
                    days.append(day_of_week.cpu().numpy())

                except Exception as e:
                    logger.warning(f"Error in test batch: {e}")
                    continue

        if not predictions:
            raise RuntimeError("No test predictions generated!")

        # Concatenate results
        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)
        days = np.concatenate(days, axis=0)

        # Calculate metrics
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        pred_flat = predictions.reshape(-1)
        actual_flat = actuals.reshape(-1)

        from sklearn.metrics import r2_score
        r2 = r2_score(actual_flat, pred_flat)
        overpred_rate = np.mean(pred_flat > actual_flat) * 100

        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'overprediction_rate': float(overpred_rate)
        }

        return predictions, actuals, days, metrics

    def train(self):
        """Main training loop for complex model - Windows safe"""
        # Create output directory
        self.output_dir = self.config.create_output_folder()
        logger.info(f"Output directory: {self.output_dir}")

        # Save config
        config_path = self.config.save_config(self.output_dir)
        logger.info(f"Config saved to: {config_path}")

        # Load data
        data = self.load_data()
        self.scaler = data.get('scaler', None)

        # Create data loaders
        train_loader, val_loader, test_loader = self.create_dataloaders(data)

        # Create complex model
        model = self.create_model()

        # Create complex loss function
        if self.config.USE_COMPLEX_LOSS:
            criterion = ComplexDayOfWeekAwareLoss(config=self.config).to(self.device)
            logger.info("Using Complex Day-Aware Loss function")
        else:
            criterion = nn.MSELoss()
            logger.info("Using standard MSE Loss function")

        # Create optimizer with lower learning rate for complex model
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )

        # Create scheduler for complex model
        scheduler = None
        if self.config.USE_SCHEDULER:
            if self.config.SCHEDULER_TYPE == 'cosine_with_warmup':
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=self.config.COSINE_RESTART_PERIOD, T_mult=2
                )
            else:
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=20, T_mult=2
                )
            logger.info(f"Using {self.config.SCHEDULER_TYPE} scheduler")

        # Training loop
        logger.info("Starting Windows-safe COMPLEX MODEL training...")
        logger.info(f"Training for {self.config.NUM_EPOCHS} epochs with patience {self.config.PATIENCE}")
        start_time = time.time()

        for epoch in range(self.config.NUM_EPOCHS):
            epoch_start = time.time()

            try:
                # Train
                train_loss = self.train_epoch(model, train_loader, criterion, optimizer, epoch)
                self.train_losses.append(train_loss)

                # Validate
                val_loss = self.validate(model, val_loader, criterion)
                self.val_losses.append(val_loss)

                # Update scheduler
                if scheduler:
                    scheduler.step()

                # Check if best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    # Save best model with training history
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'config': self.config,
                        'training_history': {
                            'train_losses': self.train_losses,
                            'val_losses': self.val_losses,
                            'day_specific_losses': self.day_specific_losses
                        }
                    }, os.path.join(self.output_dir, 'models', 'best_model.pth'))
                else:
                    self.patience_counter += 1

                # Log progress
                epoch_time = time.time() - epoch_start
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS} - "
                            f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                            f"LR: {current_lr:.8f} {'[BEST]' if is_best else ''} "
                            f"(Time: {epoch_time:.1f}s, Patience: {self.patience_counter}/{self.config.PATIENCE})")

                # Memory cleanup for complex model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                # Early stopping
                if self.patience_counter >= self.config.PATIENCE:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

            except Exception as e:
                logger.error(f"Error in epoch {epoch + 1}: {e}")
                continue

        # Training completed
        total_time = time.time() - start_time
        logger.info(f"Complex model training completed in {total_time / 60:.1f} minutes")

        # Load best model for testing
        try:
            logger.info("Loading best complex model for testing...")
            checkpoint = torch.load(os.path.join(self.output_dir, 'models', 'best_model.pth'))
            model.load_state_dict(checkpoint['model_state_dict'])

            # Test model
            logger.info("Testing complex model...")
            test_predictions, test_actuals, test_days, test_metrics = self.test_model(model, test_loader)

            # Save test results
            checkpoint['test_performance'] = test_metrics
            torch.save(checkpoint, os.path.join(self.output_dir, 'models', 'best_model.pth'))

            logger.info("COMPLEX MODEL Test Performance:")
            for metric, value in test_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")

            # Generate visualizations
            logger.info("Generating visualizations for complex model...")
            try:
                # Get validation predictions
                val_predictions = []
                val_actuals = []
                val_days = []

                model.eval()
                with torch.no_grad():
                    for x_traffic, x_temporal, y in val_loader:
                        try:
                            x_traffic = x_traffic.to(self.device)
                            x_temporal = x_temporal.to(self.device)

                            output, day_of_week = model(x_traffic, x_temporal)

                            val_predictions.append(output.cpu().numpy())
                            val_actuals.append(y.numpy())
                            val_days.append(day_of_week.cpu().numpy())
                        except Exception as e:
                            logger.warning(f"Error getting validation predictions: {e}")
                            continue

                if val_predictions:
                    val_predictions = np.concatenate(val_predictions, axis=0)
                    val_actuals = np.concatenate(val_actuals, axis=0)
                    val_days = np.concatenate(val_days, axis=0)

                    # Create visualizations
                    plot_training_history(self.train_losses, self.val_losses,
                                          self.day_specific_losses, self.output_dir)

                    # Day-specific analysis
                    val_pred_flat = val_predictions.reshape(-1)
                    val_actual_flat = val_actuals.reshape(-1)
                    val_days_expanded = np.repeat(val_days, val_predictions.shape[1] * val_predictions.shape[2])

                    analyze_day_specific_performance(val_pred_flat, val_actual_flat,
                                                     val_days_expanded, self.output_dir, "Validation")

                    test_pred_flat = test_predictions.reshape(-1)
                    test_actual_flat = test_actuals.reshape(-1)
                    test_days_expanded = np.repeat(test_days, test_predictions.shape[1] * test_predictions.shape[2])

                    analyze_day_specific_performance(test_pred_flat, test_actual_flat,
                                                     test_days_expanded, self.output_dir, "Test")

                    logger.info("Complex model visualizations completed")

            except Exception as e:
                logger.warning(f"Could not generate all visualizations: {e}")

            logger.info(f"Complex model training completed! Results saved to: {self.output_dir}")
            return model, test_metrics

        except Exception as e:
            logger.error(f"Error in testing phase: {e}")
            return model, {'mae': 0, 'rmse': 0, 'r2': 0, 'overprediction_rate': 0}


def main():
    """Main function with complex architecture configuration"""
    print("=" * 80)
    print("COMPLEX DAY-AWARE TRAFFIC PREDICTION - CENTER 1600 SQUARES")
    print("Advanced Architecture with Attention Mechanisms")
    print("=" * 80)

    # Choose configuration based on your system capabilities
    # WindowsConfig: Reduced complexity for Windows compatibility
    # ComplexConfig: Full complexity if you have sufficient hardware
    config = WindowsConfig()  # Start with this, upgrade to ComplexConfig if stable

    # Print configuration summary
    complexity_info = config.get_model_complexity_info()
    print(f"Complex Model Configuration:")
    print(f"  Architecture: {complexity_info['architecture_type']}")
    print(f"  Data Path: {config.DATA_PATH}")
    print(f"  Model Complexity: {complexity_info['complexity_level']}")
    print(f"  Memory Requirements: {complexity_info['memory_requirements']}")
    print(f"  Training Configuration:")
    print(f"    - Batch size: {config.BATCH_SIZE}")
    print(f"    - Hidden dim: {config.HIDDEN_DIM}")
    print(f"    - Attention heads: {config.ATTENTION_HEADS}")
    print(f"    - Epochs: {config.NUM_EPOCHS}")
    print(f"    - Learning rate: {config.LEARNING_RATE}")
    print(f"  Advanced Features:")
    print(f"    - Multi-scale CNN: {config.USE_MULTI_SCALE}")
    print(f"    - Spatial attention: {config.USE_SPATIAL_ATTENTION}")
    print(f"    - Complex loss: {config.USE_COMPLEX_LOSS}")
    print(f"    - Advanced augmentation: {config.USE_ADVANCED_AUGMENTATION}")
    print("=" * 80)

    # Create trainer
    trainer = ComplexWindowsSafeTrainer(config)

    try:
        # Train complex model
        model, metrics = trainer.train()

        print("\n" + "=" * 80)
        print("COMPLEX MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Output directory: {trainer.output_dir}")
        print(f"Best validation loss: {trainer.best_val_loss:.6f}")
        print(f"Final Test Results:")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  RÃ‚Â²: {metrics['r2']:.4f}")
        print(f"  Overprediction rate: {metrics['overprediction_rate']:.1f}%")
        print("=" * 80)
        print("Complex Architecture Features Used:")
        print(f"  Multi-head spatial attention ({config.ATTENTION_HEADS} heads)")
        print(f"  Advanced peak detection ({config.PEAK_TYPES} types)")
        print(f"  Multi-scale CNN ({len(config.KERNEL_SIZES)} kernels)")
        print(f"  Deep LSTM layers ({config.NUM_LAYERS} layers)")
        print(f"  Complex loss function (4 components)")
        print(f"  Sophisticated augmentation")
        print("=" * 80)

        # Performance comparison suggestion
        print("To compare with your previous model:")
        print(f"  Previous plateau: ~0.041-0.043 loss")
        print(f"  Complex model: {trainer.best_val_loss:.6f} loss")
        improvement = ((0.042 - trainer.best_val_loss) / 0.042) * 100
        print(f"  Improvement: {improvement:.1f}%")

    except Exception as e:
        logger.error(f"Complex model training failed with error: {e}")
        print(f"\nComplex model training failed! Error: {e}")
        print("\nTroubleshooting for Complex Architecture:")
        print("1. GPU Memory Issues:")
        print("   - Reduce BATCH_SIZE to 2 or even 1")
        print("   - Use LightweightConfig instead of WindowsConfig")
        print("   - Set USE_CUDA=False to use CPU only")
        print("2. CPU Memory Issues:")
        print("   - Close all other applications")
        print("   - Increase virtual memory to 16GB+")
        print("   - Use simpler WindowsConfig")
        print("3. Model Complexity:")
        print("   - Reduce HIDDEN_DIM to 256")
        print("   - Reduce ATTENTION_HEADS to 2")
        print("   - Disable some features in config")
        print("\nFallback: Use your original model.py if complex version fails")
        raise


if __name__ == "__main__":
    main()