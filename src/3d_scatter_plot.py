import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting toolkit
from matplotlib import cm  # Import colormap functionalities
import numpy as np

def plot_3d_scatter(data_file):
    # Read data from Excel or CSV file
    df = pd.read_excel(data_file)  # If using CSV, replace with pd.read_csv(data_file)

    # Extract the required columns
    cci = df['Color Cast Index']  # Color Cast Index
    dpp = df['Dark Pixel Percentage (%)']  # Dark Pixel Percentage
    blur = df['Blur Amount']  # Blur Amount
    filenames = df['Image Name']  # Image names (optional for labeling)

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the scatter points with color mapping based on the blur amount
    scatter = ax.scatter(cci, dpp, blur, c=blur, cmap='viridis', s=50, alpha=0.8)

    # Add a color bar indicating the blur amount
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Blur Amount')

    # Set labels for the axes
    ax.set_xlabel('Color Cast Index', fontsize=12)
    ax.set_ylabel('Dark Pixel Percentage (%)', fontsize=12)
    ax.set_zlabel('Blur Amount', fontsize=12)

    # Set the size of tick labels
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Set the title of the plot
    ax.set_title('3D Scatter Plot of Underwater Image Metrics', fontsize=16)

    # Adjust the viewing angle
    ax.view_init(elev=20, azim=45)  # Adjust elevation and azimuth as needed

    # Display grid lines
    ax.grid(True)

    # Show the plot
    plt.show()


# Example usage
data_file = '../../result/result3.xlsx'  # Replace with your actual data file
plot_3d_scatter(data_file)
