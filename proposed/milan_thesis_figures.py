import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import pickle
import warnings
from datetime import datetime, timedelta
import os
from scipy import stats

warnings.filterwarnings('ignore')

# Set publication-quality plot parameters
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1


class ThesisFigureGenerator:
    def __init__(self, data_path, output_dir='figures'):
        """
        Initialize the figure generator for thesis

        Args:
            data_path: Path to preprocessed data pickle file
            output_dir: Directory to save figures
        """
        self.data_path = data_path
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Load the preprocessed data
        print(f"Loading data from: {data_path}")
        with open(data_path, 'rb') as f:
            self.data_dict = pickle.load(f)

        print("Data loaded successfully!")
        self._prepare_data()

    def _prepare_data(self):
        """Prepare data for analysis"""
        # Extract training data for time series analysis
        self.X_traffic_train = self.data_dict['X_traffic_train']
        self.X_temporal_train = self.data_dict['X_temporal_train']
        self.y_train = self.data_dict['y_train']

        # Also get validation and test data
        self.X_traffic_val = self.data_dict['X_traffic_val']
        self.X_temporal_val = self.data_dict['X_temporal_val']

        self.X_traffic_test = self.data_dict['X_traffic_test']
        self.X_temporal_test = self.data_dict['X_temporal_test']

        # Create a continuous time series from the training data
        # Reshape to get average traffic across all squares
        train_traffic_flat = self.X_traffic_train.reshape(-1, 1600)
        self.avg_traffic = train_traffic_flat.mean(axis=1)

        # Extract temporal features
        self._extract_temporal_info()

        print(f"Data shapes:")
        print(f"  Training samples: {self.X_traffic_train.shape[0]}")
        print(f"  Time series length: {len(self.avg_traffic)}")

    def _extract_temporal_info(self):
        """Extract hour and day information from temporal features"""
        # Get all temporal data
        all_temporal = np.vstack([
            self.X_temporal_train.reshape(-1, 11),
            self.X_temporal_val.reshape(-1, 11),
            self.X_temporal_test.reshape(-1, 11)
        ])

        all_traffic = np.vstack([
            self.X_traffic_train.reshape(-1, 1600),
            self.X_traffic_val.reshape(-1, 1600),
            self.X_traffic_test.reshape(-1, 1600)
        ])

        # Day of week is in first 7 features (one-hot)
        self.day_of_week = np.argmax(all_temporal[:, :7], axis=1)

        # Hour is in feature 7
        self.hour = all_temporal[:, 7].astype(int)

        # Average traffic for each sample
        self.all_avg_traffic = all_traffic.mean(axis=1)

        print(f"Temporal data extracted: {len(self.day_of_week)} samples")

    def generate_acf_pacf_plots(self, max_lags=1200):
        """
        Generate ACF and PACF plots (Figure 2.2)

        Args:
            max_lags: Maximum number of lags to display
        """
        print("\nGenerating ACF and PACF plots...")

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # ACF plot
        plot_acf(self.avg_traffic, lags=max_lags, ax=ax1, alpha=0.05)
        ax1.set_title('(a) Autocorrelation Function', fontsize=18, pad=10)
        ax1.set_xlabel('Lag (10-minute intervals)', fontsize=18)
        ax1.set_ylabel('Autocorrelation', fontsize=18)
        ax1.grid(True, alpha=0.3, linestyle='--')



        ax1.axvline(x=1008, color='blue', linestyle='--', alpha=0.5, linewidth=0.8)
        ax1.text(1008, ax1.get_ylim()[1] * 0.9, 'Weekly\n(1008 lags)',
                 ha='center', fontsize=15, color='red')

        # PACF plot (with fewer lags for clarity)
        plot_pacf(self.avg_traffic, lags=min(300, max_lags // 4), ax=ax2, alpha=0.05)
        ax2.set_title('(b) Partial Autocorrelation Function', fontsize=18, pad=10)
        ax2.set_xlabel('Lag (10-minute intervals)', fontsize=18)
        ax2.set_ylabel('Partial Autocorrelation', fontsize=18)
        ax2.grid(True, alpha=0.3, linestyle='--')


        # --- ACF daily annotation (move to bottom) ---
        ax1.axvline(x=144, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
        ax1.text(144, 0.05, 'Daily\n(144 lags)',
                 transform=ax1.get_xaxis_transform(),  # x in data, y in [0,1]
                 ha='center', va='bottom', fontsize=15, color='red')


        # --- PACF daily annotation (move to bottom) ---
        ax2.axvline(x=144, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
        ax2.text(144, 0.05, 'Daily\n(144 lags)',
                 transform=ax2.get_xaxis_transform(),
                 ha='center', va='bottom', fontsize=15, color='red')

        plt.tight_layout()

        # Save figure
        acf_path = os.path.join(self.output_dir, 'acf_pacf.pdf')
        plt.savefig(acf_path, format='pdf', bbox_inches='tight')
        print(f"ACF/PACF plot saved to: {acf_path}")

        # Also save as PNG for quick viewing
        plt.savefig(acf_path.replace('.pdf', '.png'), format='png', dpi=150)
        plt.show()

    def generate_peak_traffic_histogram(self, peak_percentile=90):
        """
        Generate peak traffic distribution histogram (Figure 2.3)

        Args:
            peak_percentile: Percentile threshold for peak traffic
        """
        print(f"\nGenerating peak traffic distribution (>{peak_percentile}th percentile)...")

        # Calculate peak threshold
        threshold = np.percentile(self.all_avg_traffic, peak_percentile)
        print(f"Peak threshold: {threshold:.4f}")

        # Identify peak events
        peak_mask = self.all_avg_traffic > threshold
        peak_hours = self.hour[peak_mask]
        peak_days = self.day_of_week[peak_mask]

        # Create figure with two subplots
        fig = plt.figure(figsize=(14, 6))

        # Subplot 1: Peak events by hour
        ax1 = plt.subplot(1, 2, 1)
        hour_counts, hour_bins, _ = ax1.hist(peak_hours, bins=24, range=(0, 24),
                                             color='steelblue', edgecolor='black',
                                             alpha=0.7, linewidth=0.5)
        ax1.set_xlabel('Hour of Day', fontsize=11)
        ax1.set_ylabel('Number of Peak Events', fontsize=11)
        ax1.set_title('(a) Peak Traffic Events by Hour', fontsize=12, pad=10)
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax1.set_xticks(range(0, 24, 2))

        # Highlight rush hours
        morning_rush = (8, 10)
        evening_rush = (17, 20)
        ax1.axvspan(morning_rush[0], morning_rush[1], alpha=0.2, color='orange',
                    label='Morning rush')
        ax1.axvspan(evening_rush[0], evening_rush[1], alpha=0.2, color='red',
                    label='Evening rush')
        ax1.legend(loc='upper right', fontsize=9)

        # Subplot 2: Peak events by day of week
        ax2 = plt.subplot(1, 2, 2)
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_counts = np.bincount(peak_days, minlength=7)

        bars = ax2.bar(range(7), day_counts, color='darkgreen', alpha=0.7,
                       edgecolor='black', linewidth=0.5)

        # Color weekends differently
        bars[5].set_color('coral')
        bars[6].set_color('coral')

        ax2.set_xlabel('Day of Week', fontsize=11)
        ax2.set_ylabel('Number of Peak Events', fontsize=11)
        ax2.set_title('(b) Peak Traffic Events by Day', fontsize=12, pad=10)
        ax2.set_xticks(range(7))
        ax2.set_xticklabels(day_names)
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

        # Add legend for weekend highlighting
        from matplotlib.patches import Patch
        weekday_patch = Patch(color='darkgreen', alpha=0.7, label='Weekday')
        weekend_patch = Patch(color='coral', alpha=0.7, label='Weekend')
        ax2.legend(handles=[weekday_patch, weekend_patch], loc='upper right', fontsize=9)

        plt.suptitle(f'Distribution of Peak Traffic Events (>{peak_percentile}th Percentile)',
                     fontsize=13, y=1.02)
        plt.tight_layout()

        # Save figure
        peak_path = os.path.join(self.output_dir, 'peak_histogram.pdf')
        plt.savefig(peak_path, format='pdf', bbox_inches='tight')
        print(f"Peak traffic histogram saved to: {peak_path}")

        # Also save as PNG
        plt.savefig(peak_path.replace('.pdf', '.png'), format='png', dpi=150)
        plt.show()

        # Print statistics
        total_samples = len(self.all_avg_traffic)
        peak_samples = np.sum(peak_mask)
        peak_volume = np.sum(self.all_avg_traffic[peak_mask])
        total_volume = np.sum(self.all_avg_traffic)

        print(f"\nPeak Traffic Statistics:")
        print(f"  Total samples: {total_samples:,}")
        print(f"  Peak samples: {peak_samples:,} ({peak_samples / total_samples * 100:.1f}%)")
        print(f"  Peak traffic volume: {peak_volume / total_volume * 100:.1f}% of total")

    def generate_additional_analysis_plots(self):
        """Generate additional analysis plots for deeper insights"""
        print("\nGenerating additional analysis plots...")

        # Create a 2x2 subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))  # Much larger figure

        # 1. Traffic heatmap by hour and day
        ax = axes[0, 0]
        traffic_pivot = pd.DataFrame({
            'hour': self.hour,
            'day': self.day_of_week,
            'traffic': self.all_avg_traffic
        }).pivot_table(values='traffic', index='hour', columns='day', aggfunc='mean')

        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        sns.heatmap(traffic_pivot, cmap='YlOrRd', ax=ax,
                    xticklabels=day_names, cbar_kws={'label': 'Avg Traffic'})
        ax.set_title('(a) Average Traffic Intensity Heatmap', fontsize=22)  # Much larger
        ax.set_xlabel('Day of Week', fontsize=20)  # Much larger
        ax.set_ylabel('Hour of Day', fontsize=20)  # Much larger
        ax.tick_params(axis='both', labelsize=18)  # Ensure tick labels are large

        # 2. Traffic distribution
        ax = axes[0, 1]
        ax.hist(self.all_avg_traffic, bins=50, density=True,
                alpha=0.7, color='steelblue', edgecolor='black', linewidth=1.0)

        # Fit and plot distributions
        # Fit normal distribution
        mu, std = stats.norm.fit(self.all_avg_traffic)
        x = np.linspace(self.all_avg_traffic.min(), self.all_avg_traffic.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2.5,
                label=f'Normal fit\n(μ={mu:.3f}, σ={std:.3f})')

        ax.set_xlabel('Traffic Intensity (Normalized)', fontsize=20)  # Much larger
        ax.set_ylabel('Density', fontsize=20)  # Much larger
        ax.set_title('(b) Traffic Intensity Distribution', fontsize=22)  # Much larger
        ax.legend(loc='upper right', fontsize=16)  # Much larger
        ax.tick_params(axis='both', labelsize=18)  # Ensure tick labels are large
        ax.grid(True, alpha=0.3, linestyle='--')

        # 3. Weekly pattern
        ax = axes[1, 0]
        # Create a week's worth of data
        week_length = min(1008, len(self.avg_traffic))  # One week
        week_data = self.avg_traffic[:week_length]
        time_axis = np.arange(week_length) / 144  # Convert to days

        ax.plot(time_axis, week_data, linewidth=1.5, color='navy', alpha=0.7)
        ax.set_xlabel('Days', fontsize=20)  # Much larger
        ax.set_ylabel('Average Traffic', fontsize=20)  # Much larger
        ax.set_title('(c) Weekly Traffic Pattern', fontsize=22)  # Much larger
        ax.tick_params(axis='both', labelsize=18)  # Ensure tick labels are large
        ax.grid(True, alpha=0.3, linestyle='--')

        # Mark day boundaries
        for day in range(1, 8):
            ax.axvline(x=day, color='gray', linestyle=':', alpha=0.5, linewidth=1.0)

        # 4. Quantile analysis
        ax = axes[1, 1]
        quantiles = [10, 25, 50, 75, 90, 95, 99]
        quantile_values = [np.percentile(self.all_avg_traffic, q) for q in quantiles]

        ax.bar(range(len(quantiles)), quantile_values,
               color='darkgreen', alpha=0.7, edgecolor='black', linewidth=1.0)
        ax.set_xticks(range(len(quantiles)))
        ax.set_xticklabels([f'{q}th' for q in quantiles], fontsize=18)  # Explicitly set size
        ax.set_xlabel('Percentile', fontsize=20)  # Much larger
        ax.set_ylabel('Traffic Intensity', fontsize=20)  # Much larger
        ax.set_title('(d) Traffic Intensity Percentiles', fontsize=22)  # Much larger
        ax.tick_params(axis='y', labelsize=18)  # Ensure y-tick labels are large
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        # Highlight extreme percentiles
        ax.bar([5, 6], [quantile_values[5], quantile_values[6]],
               color='red', alpha=0.7, edgecolor='black', linewidth=1.0)

        plt.suptitle('Traffic Pattern Analysis', fontsize=24, y=1.02)  # Much larger
        plt.tight_layout()

        # Save figure
        analysis_path = os.path.join(self.output_dir, 'traffic_analysis.pdf')
        plt.savefig(analysis_path, format='pdf', bbox_inches='tight')
        print(f"Additional analysis plot saved to: {analysis_path}")

        # Also save as PNG
        plt.savefig(analysis_path.replace('.pdf', '.png'), format='png', dpi=150)
        plt.show()

    def generate_all_figures(self):
        """Generate all thesis figures"""
        print("=" * 60)
        print("GENERATING ALL THESIS FIGURES")
        print("=" * 60)

        # Generate ACF/PACF plots (Figure 2.2)
        self.generate_acf_pacf_plots(max_lags=1200)

        # Generate peak traffic histogram (Figure 2.3)
        self.generate_peak_traffic_histogram(peak_percentile=90)

        # Generate additional analysis plots
        self.generate_additional_analysis_plots()

        print("\n" + "=" * 60)
        print("ALL FIGURES GENERATED SUCCESSFULLY!")
        print("=" * 60)
        print(f"\nFigures saved in: {os.path.abspath(self.output_dir)}")
        print("\nGenerated files:")
        for file in os.listdir(self.output_dir):
            if file.endswith(('.pdf', '.png')):
                file_path = os.path.join(self.output_dir, file)
                size_kb = os.path.getsize(file_path) / 1024
                print(f"  - {file} ({size_kb:.1f} KB)")


def main():
    """Main function to generate thesis figures"""

    # Set your data path - USE RAW STRING
    data_path = r"C:\Users\Fan\PycharmProjects\Msc_Thesis\processed_data\preprocessed_milan_traffic_center_1600_7day_splits_optimized.pkl"

    # Output directory for figures
    output_dir = r"C:\Users\Fan\PycharmProjects\Msc_Thesis\figures"

    # Create figure generator
    generator = ThesisFigureGenerator(data_path, output_dir)

    # Generate all figures
    generator.generate_all_figures()

    # Print LaTeX code for including figures
    print("\n" + "=" * 60)
    print("LaTeX CODE FOR INCLUDING FIGURES IN THESIS:")
    print("=" * 60)
    print("""
% Figure 2.2: ACF and PACF
\\begin{figure}[htbp]
    \\centering
    \\includegraphics[width=\\textwidth]{figures/acf_pacf.pdf}
    \\caption{Temporal correlation structure of Milan traffic data showing 
             (a) autocorrelation function and (b) partial autocorrelation function. 
             Strong periodicities are evident at 144 lags (daily) and 1,008 lags (weekly).}
    \\label{fig:acf_pacf}
\\end{figure}

% Figure 2.3: Peak Traffic Distribution
\\begin{figure}[htbp]
    \\centering
    \\includegraphics[width=\\textwidth]{figures/peak_histogram.pdf}
    \\caption{Distribution of peak traffic events (>90th percentile) by 
             (a) hour of day showing morning (8-10 AM) and evening (5-8 PM) rush hours, and 
             (b) day of week revealing higher weekday peak frequencies.}
    \\label{fig:peak_distribution}
\\end{figure}

% Additional Analysis Figure
\\begin{figure}[htbp]
    \\centering
    \\includegraphics[width=\\textwidth]{figures/traffic_analysis.pdf}
    \\caption{Comprehensive traffic pattern analysis: (a) hourly intensity heatmap, 
             (b) traffic distribution with normal fit, (c) weekly pattern, and 
             (d) percentile analysis highlighting extreme values.}
    \\label{fig:traffic_analysis}
\\end{figure}
    """)


if __name__ == "__main__":
    main()