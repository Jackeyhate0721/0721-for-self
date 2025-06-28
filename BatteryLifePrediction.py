# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')


# ==================== 1. Configuration ====================
def set_visualization():
    """Configure matplotlib settings"""
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'axes.unicode_minus': False,
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'axes.labelsize': 11,
        'axes.titlesize': 12
    })
    sns.set_palette("viridis")
    print("Visualization settings configured")


set_visualization()


# ==================== 2. Data Loading ====================
def load_battery_data(filepath):
    """Load battery data from CSV file"""
    df = pd.read_csv(filepath)

    # Rename columns to match our analysis framework
    df = df.rename(columns={
        'cycle': 'Cycle',
        'capacity': 'Capacity(Ah)',
        'SoH': 'SoH',
        'resistance': 'Resistance(Ohm)',
        'CCCT': 'CC_Time(s)',
        'CVCT': 'CV_Time(s)'
    })

    # Add temperature column (assuming constant temperature for this dataset)
    df['Temperature(°C)'] = 25

    # Create temperature groups
    bins = 1  # Only one temperature group for this dataset
    labels = ["25°C"]
    df['Temp_Group'] = pd.cut(df['Temperature(°C)'], bins=bins, labels=labels)

    return df


# ==================== 3. Data Cleaning ====================
def clean_data(df):
    """Clean the battery data"""
    # Basic filtering
    df = df[
        (df['Capacity(Ah)'] >= df['Capacity(Ah)'].min() * 0.8) &  # At least 80% of min capacity
        (df['Capacity(Ah)'] <= df['Capacity(Ah)'].max() * 1.05)  # Up to 105% of max capacity
        ].copy()

    # Sort by cycle
    df = df.sort_values('Cycle')

    # Calculate capacity change between cycles
    df['cap_change'] = df['Capacity(Ah)'].diff().abs()

    # Remove outliers in capacity change
    threshold = df['cap_change'].quantile(0.99)  # Remove top 1% of changes
    df = df[(df['cap_change'] <= threshold) | df['cap_change'].isna()]

    return df.drop(columns=['cap_change'], errors='ignore')


# ==================== 4. Analysis Core ====================
def perform_degradation_analysis(df):
    """Perform degradation analysis on the battery data"""
    # Linear regression for capacity fade
    x = df['Cycle'].values
    y = df['Capacity(Ah)'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Calculate SoH degradation rate
    soh_slope, soh_intercept, soh_r_value, _, _ = stats.linregress(x, df['SoH'].values)

    # Calculate cycles to 80% SoH (assuming linear degradation)
    eol_cycle = (0.8 - soh_intercept) / soh_slope if soh_slope < 0 else float('inf')

    return {
        'capacity_slope': slope,
        'capacity_intercept': intercept,
        'capacity_r2': r_value ** 2,
        'soh_slope': soh_slope,
        'soh_intercept': soh_intercept,
        'soh_r2': soh_r_value ** 2,
        'eol_cycle': eol_cycle,
        'std_err': std_err
    }


# ==================== 5. Visualizations ====================
def plot_capacity_degradation(df):
    """Plot capacity degradation over cycles"""
    plt.figure(figsize=(10, 6))
    plt.plot(df['Cycle'], df['Capacity(Ah)'], 'b-', label='Capacity')
    plt.xlabel('Cycle Number')
    plt.ylabel('Capacity (Ah)')
    plt.title('Battery Capacity Degradation')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('capacity_degradation.png')
    plt.close()


def plot_soh_degradation(df):
    """Plot State of Health degradation over cycles"""
    plt.figure(figsize=(10, 6))
    plt.plot(df['Cycle'], df['SoH'], 'r-', label='SoH')
    plt.axhline(y=0.8, color='gray', linestyle='--', label='80% SoH Threshold')
    plt.xlabel('Cycle Number')
    plt.ylabel('State of Health (SoH)')
    plt.title('Battery State of Health Degradation')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('soh_degradation.png')
    plt.close()


def plot_resistance(df):
    """Plot internal resistance over cycles"""
    plt.figure(figsize=(10, 6))
    plt.plot(df['Cycle'], df['Resistance(Ohm)'], 'g-', label='Internal Resistance')
    plt.xlabel('Cycle Number')
    plt.ylabel('Resistance (Ohm)')
    plt.title('Battery Internal Resistance Change')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('resistance_change.png')
    plt.close()


def plot_charging_time(df):
    """Plot charging times over cycles"""
    plt.figure(figsize=(10, 6))
    plt.plot(df['Cycle'], df['CC_Time(s)'], 'b-', label='CC Time')
    plt.plot(df['Cycle'], df['CV_Time(s)'], 'r-', label='CV Time')
    plt.xlabel('Cycle Number')
    plt.ylabel('Time (seconds)')
    plt.title('Charging Time Evolution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('charging_time.png')
    plt.close()


def create_correlation_matrix(df):
    """Create correlation matrix of all parameters"""
    corr_df = df[['Capacity(Ah)', 'SoH', 'Resistance(Ohm)', 'CC_Time(s)', 'CV_Time(s)']].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Parameter Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()


def create_battery_report(df, results):
    """Create comprehensive battery report"""
    plt.figure(figsize=(12, 8))

    # Create subplots
    ax1 = plt.subplot2grid((3, 2), (0, 0))
    ax2 = plt.subplot2grid((3, 2), (0, 1))
    ax3 = plt.subplot2grid((3, 2), (1, 0))
    ax4 = plt.subplot2grid((3, 2), (1, 1))
    ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)

    # Plot capacity
    ax1.plot(df['Cycle'], df['Capacity(Ah)'], 'b-')
    ax1.set_title('Capacity Degradation')
    ax1.set_xlabel('Cycle')
    ax1.set_ylabel('Capacity (Ah)')
    ax1.grid(True, alpha=0.3)

    # Plot SoH
    ax2.plot(df['Cycle'], df['SoH'], 'r-')
    ax2.axhline(0.8, color='gray', linestyle='--')
    ax2.set_title('State of Health')
    ax2.set_xlabel('Cycle')
    ax2.set_ylabel('SoH')
    ax2.grid(True, alpha=0.3)

    # Plot resistance
    ax3.plot(df['Cycle'], df['Resistance(Ohm)'], 'g-')
    ax3.set_title('Internal Resistance')
    ax3.set_xlabel('Cycle')
    ax3.set_ylabel('Resistance (Ohm)')
    ax3.grid(True, alpha=0.3)

    # Plot charging times
    ax4.plot(df['Cycle'], df['CC_Time(s)'], 'b-', label='CC Time')
    ax4.plot(df['Cycle'], df['CV_Time(s)'], 'r-', label='CV Time')
    ax4.set_title('Charging Times')
    ax4.set_xlabel('Cycle')
    ax4.set_ylabel('Time (s)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add text report
    report_text = f"""
=== BATTERY DEGRADATION REPORT ===
Initial Capacity: {df['Capacity(Ah)'].iloc[0]:.3f} Ah
Final Capacity: {df['Capacity(Ah)'].iloc[-1]:.3f} Ah
Capacity Loss: {100 * (df['Capacity(Ah)'].iloc[0] - df['Capacity(Ah)'].iloc[-1]) / df['Capacity(Ah)'].iloc[0]:.1f}%
Capacity Degradation Rate: {results['capacity_slope']:.2e} Ah/cycle (R²={results['capacity_r2']:.2f})
SoH Degradation Rate: {results['soh_slope']:.2e} SoH/cycle (R²={results['soh_r2']:.2f})
Predicted EOL (80% SoH): {results['eol_cycle']:.0f} cycles
Total Cycles Analyzed: {len(df)}
"""
    ax5.text(0.05, 0.1, report_text, fontfamily='monospace', fontsize=10)
    ax5.axis('off')

    plt.tight_layout()
    plt.savefig('battery_report.png', dpi=300)
    plt.close()


# ==================== 6. Main Program ====================
if __name__ == "__main__":
    print("=== BATTERY DATA ANALYSIS ===")

    try:
        # 1. Load the CSV file
        print("\n1. Loading battery data...")
        filepath = 'CS2_35.csv'
        raw_data = load_battery_data(filepath)
        raw_data.to_csv('raw_battery_data.csv', index=False)

        # 2. Clean the data
        print("2. Cleaning data...")
        cleaned_data = clean_data(raw_data)
        cleaned_data.to_csv('cleaned_battery_data.csv', index=False)

        # 3. Perform analysis
        print("3. Running analysis...")
        results = perform_degradation_analysis(cleaned_data)

        # 4. Generate visualizations
        print("4. Generating visualizations...")
        plot_capacity_degradation(cleaned_data)
        plot_soh_degradation(cleaned_data)
        plot_resistance(cleaned_data)
        plot_charging_time(cleaned_data)
        create_correlation_matrix(cleaned_data)
        create_battery_report(cleaned_data, results)

        print("\n=== ANALYSIS COMPLETED ===")
        print(f"Capacity Degradation Rate: {results['capacity_slope']:.2e} Ah/cycle")
        print(f"SoH Degradation Rate: {results['soh_slope']:.2e} SoH/cycle")
        print(f"Predicted EOL (80% SoH): {results['eol_cycle']:.0f} cycles")

        print("\nGenerated Files:")
        print("- raw_battery_data.csv")
        print("- cleaned_battery_data.csv")
        print("- capacity_degradation.png")
        print("- soh_degradation.png")
        print("- resistance_change.png")
        print("- charging_time.png")
        print("- correlation_matrix.png")
        print("- battery_report.png")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Troubleshooting:")
        print("1. Make sure CS2_35.csv is in the same directory")
        print("2. Install required packages: pip install numpy scipy matplotlib seaborn pandas")
        exit(1)