import pandas as pd
import numpy as np


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

def plot_histogram_with_subtracted_mean(df, column_name, name, num_bins):
    """
    Given a DataFrame and a column name, this function calculates a wide range of statistics 
    (mean, std, min, max, count, skewness, kurtosis), subtracts the mean from the values of the 
    specified column, and plots a histogram of the new values.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column to analyze.

    Returns:
    None
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame")
    
    # Step 1: Calculate Statistics
    stats = {
        'Mean': df[column_name].mean(),
        'Standard Deviation': df[column_name].std(),
        'Min': df[column_name].min(),
        'Max': df[column_name].max(),
        'Count': df[column_name].count(),
        'Skewness': skew(df[column_name], nan_policy='omit'),
        'Kurtosis': kurtosis(df[column_name], nan_policy='omit'),
        'Median': df[column_name].median(),
        '25th Percentile (Q1)': df[column_name].quantile(0.25),
        '50th Percentile (Q2 / Median)': df[column_name].quantile(0.5),
        '75th Percentile (Q3)': df[column_name].quantile(0.75),
        'Interquartile Range (IQR)': df[column_name].quantile(0.75) - df[column_name].quantile(0.25),
        'Variance': df[column_name].var()
    }

    # Print out the statistics
    print(f"Statistics for column '{column_name}':")
    for stat, value in stats.items():
        print(f"{stat}: {value}")

    # Step 2: Get corresponding `X` and `Y` for min and max values in the original column
    min_value = stats['Min']
    max_value = stats['Max']

    # Find the rows where the column value equals the min and max
    min_value_rows = df[df[column_name] == min_value]
    max_value_rows = df[df[column_name] == max_value]

    # Assuming 'X' and 'Y' are columns in the dataframe, otherwise replace them with actual column names
    min_value_x_y = min_value_rows[['X', 'Y']] if 'X' in df.columns and 'Y' in df.columns else None
    max_value_x_y = max_value_rows[['X', 'Y']] if 'X' in df.columns and 'Y' in df.columns else None

    # Print the rows corresponding to the min and max values
    print(f"Rows corresponding to the minimum value ({min_value}):")
    print(min_value_x_y)
    
    print(f"Rows corresponding to the maximum value ({max_value}):")
    print(max_value_x_y)


    # Step 2: Subtract the mean from the column
    df[column_name + '_subtracted'] = df[column_name] - stats['Mean']

    # Step 3: Plot the Histogram
    plt.figure(figsize=(8,6))

    # Calculate the min and max of the data for the new column
    min_value = df[column_name + '_subtracted'].min()
    max_value = df[column_name + '_subtracted'].max()

    # Plot histogram with dynamic number of bins and focus on data range
    plt.hist(df[column_name + '_subtracted'], bins=num_bins, edgecolor='black', alpha=0.7)
    
    # Set the limits to focus on the data region
    plt.xlim(min_value, max_value)

    # Adding titles and labels
    plt.title(f'Histogram of {column_name} after Subtracting Mean')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f'./{name}.png')



def plot_histogram_diff_dist(column, name, num_bins):
    """
    Given a pandas Series (single column), this function calculates a wide range of statistics
    (mean, std, min, max, count, skewness, kurtosis), subtracts the mean from the values of the
    specified column, and plots a histogram of the new values.

    Parameters:
    column (pd.Series): The input column (single pandas Series).
    name (str): The name for the plot (used in the file name).
    num_bins (int): The number of bins for the histogram.

    Returns:
    None
    """
    # Check if the input is a pandas Series
    if not isinstance(column, pd.Series):
        raise ValueError("The input must be a pandas Series.")
    
    # Step 1: Calculate Statistics
    stats = {
        'Mean': column.mean(),
        'Standard Deviation': column.std(),
        'Min': column.min(),
        'Max': column.max(),
        'Count': column.count(),
        'Skewness': skew(column, nan_policy='omit'),
        'Kurtosis': kurtosis(column, nan_policy='omit'),
        'Median': column.median(),
        '25th Percentile (Q1)': column.quantile(0.25),
        '50th Percentile (Q2 / Median)': column.quantile(0.5),
        '75th Percentile (Q3)': column.quantile(0.75),
        'Interquartile Range (IQR)': column.quantile(0.75) - column.quantile(0.25),
        'Variance': column.var()
    }

    # Print out the statistics
    print(f"Statistics for the column:")
    for stat, value in stats.items():
        print(f"{stat}: {value}")

    # Step 2: Subtract the mean from the column
    column_subtracted = column - stats['Mean']

    # Step 3: Plot the Histogram
    plt.figure(figsize=(8,6))

    # Calculate the min and max of the data for the new column
    min_value = column_subtracted.min()
    max_value = column_subtracted.max()

    # Plot histogram with dynamic number of bins and focus on data range
    plt.hist(column_subtracted, bins=num_bins, edgecolor='black', alpha=0.7)
    
    # Set the limits to focus on the data region
    plt.xlim(-30, 30)

    # Setting x-axis ticks with 50 increments
    x_ticks = range(int(-30 // 10) * 10, int(30 // 10) * 10 + 10, 10)
    plt.xticks(x_ticks)
    # Adding titles and labels
    plt.title(f'Histogram after Subtracting Mean')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Save the plot as an image file
    plt.savefig(f'./{name}.png')


# noisy = pd.read_csv('/home/santosh/Projects/geo_physics/meixia_noise_removal/brazil_v2_data/airmag_recent_MAGCOR_IGRF.csv')
# gt = pd.read_csv('/home/santosh/Projects/geo_physics/meixia_noise_removal/brazil_v2_data/airmag_recent_MAGIGRF.csv')
# denoised = pd.read_csv('/home/santosh/Projects/geo_physics/meixia_noise_removal/pretraining/final_brazil_runs/finetuning_256_5e5_l1loss_maintain_same_patch_also/2024-12-10_10-11-14/airmag/maintain_same_also_output_v3_no_fft.csv')
# noisy_gt_dist = noisy['MAGCOR_IGRF'] - gt['MAGIGRF']
# noisy_denoised_dist = noisy['MAGCOR_IGRF'] - denoised['MAGIGRF']

# # plot_histogram_with_subtracted_mean(noisy, 'MAGCOR_IGRF', name = 'airmag_noisy_hist', num_bins=100)
# # plot_histogram_with_subtracted_mean(gt, 'MAGIGRF', name = 'airmag_clean_hist', num_bins=100)
# # plot_histogram_with_subtracted_mean(denoised, 'MAGIGRF', name = 'airmag_denoised_hist', num_bins=100)
# plot_histogram_diff_dist(noisy_gt_dist, name = 'airmag_clean_diff_hist', num_bins = 200)
# plot_histogram_diff_dist(noisy_denoised_dist, name = 'airmag_denoised_diff_hist', num_bins = 200)

noisy = pd.read_csv('/home/santosh/Projects/geo_physics/meixia_noise_removal/brazil_v2_data/test1_1111_MAGCOR_IGRF.csv')
gt = pd.read_csv('/home/santosh/Projects/geo_physics/meixia_noise_removal/brazil_v2_data/test1_1111_MAGIGRF.csv')
denoised = pd.read_csv('/home/santosh/Projects/geo_physics/meixia_noise_removal/pretraining/final_brazil_runs/finetuning_256_5e5_l1loss_maintain_same_patch_also/2024-12-10_10-11-14/airmag/brazil_maintain_same_also_output_v3_no_fft.csv')

noisy_gt_dist = noisy['MAGCOR_IGRF'] - gt['MAGIGRF']
noisy_denoised_dist = noisy['MAGCOR_IGRF'] - denoised['MAGIGRF']
plot_histogram_diff_dist(noisy_gt_dist, name = 'brazil_clean_diff_hist', num_bins = 200)
plot_histogram_diff_dist(noisy_denoised_dist, name = 'brazil_denoised_diff_hist', num_bins = 200)