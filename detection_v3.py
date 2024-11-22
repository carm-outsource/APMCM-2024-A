import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting toolkit
from matplotlib import cm  # Import colormap functionalities

# Constants
DATASET_FOLDER = 'resources/a/'  # Replace with your image folder path
OUTPUT_EXCEL = 'result/result4.xlsx'  # Output xlsx file name
OUTPUT_IMAGE = 'result/plot4.png'  # Output image file name

# Read image files with these extensions
IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']

# Set threshold
CCI_THRESHOLD = 15
DARK_PIXEL_THRESHOLD = 50
EDGE_DENSITY_THRESHOLD = 5  # Adjust based on actual situation

BLUR_THRESHOLD = 600  # Adjust based on actual situation
STD_THRESHOLD = 10  # Adjust based on actual situation


class ColorMetrics:

    def __init__(self, offset):
        self.offset = offset

    def result(self):
        return self.offset > CCI_THRESHOLD


class LightMetrics:

    def __init__(self, dark_pixel_percentage, mean_brightness, brightness_std_dev, entropy):
        self.dark_pixel_percentage = dark_pixel_percentage
        self.mean_brightness = mean_brightness
        self.brightness_std_dev = brightness_std_dev
        self.entropy = entropy

    def result(self):
        return self.dark_pixel_percentage > 40 and self.mean_brightness < 80 and self.entropy < 7


class BlurMetrics:
    def __init__(self, blur, edge_density):
        self.blur = blur
        self.edge_density = edge_density

    def result(self):
        if self.edge_density >= EDGE_DENSITY_THRESHOLD:
            return self.blur < BLUR_THRESHOLD
        else:
            return self.blur < STD_THRESHOLD


def color_metric(img):
    _, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    return ColorMetrics(np.sqrt((np.mean(a) - 128) ** 2 + (np.mean(b) - 128) ** 2))


def light_metric(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    dark_pixel_size = np.sum(gray < DARK_PIXEL_THRESHOLD)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    hist_norm = hist_norm + 1e-7  # Avoid log(0)
    return LightMetrics(
        (dark_pixel_size / gray.size) * 100,
        np.mean(gray), np.std(gray),
        -np.sum(hist_norm * np.log2(hist_norm))
    )


def blur_metric(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate edge density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = (np.sum(edges > 0) / gray.size) * 100

    if edge_density >= EDGE_DENSITY_THRESHOLD:
        # High edge density, use Tenengrad method
        Gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        Gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        blur = np.mean(Gx ** 2 + Gy ** 2)
    else:
        # Low edge density, use gray level variance
        blur = np.std(gray)

    return BlurMetrics(blur, edge_density)


# Process images in the folder
def process_images(folder_path, output_excel, output_image):
    results = []
    for filename in os.listdir(folder_path):

        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            continue  # Check if the file is an image

        img = cv2.imread(os.path.join(folder_path, filename))

        if img is None:
            print(f"Cannot read image: {filename}")
            continue

        print(f"Processing: {filename} ...")
        color_metrics = color_metric(img)
        light_metrics = light_metric(img)
        blur_metrics = blur_metric(img)

        # Add results to list
        results.append({
            'Image Name': filename,
            'COLOR': color_metrics.result(),
            'LIGHT': light_metrics.result(),
            'BLUR': blur_metrics.result(),
            'Color Cast Index': color_metrics.offset,
            'Dark Pixel Percentage (%)': light_metrics.dark_pixel_percentage,
            'Mean Brightness': light_metrics.mean_brightness,
            'Brightness Standard Deviation': light_metrics.brightness_std_dev,
            'Entropy': light_metrics.entropy,
            'Blur Amount': blur_metrics.blur,
            'Edge Density (%)': blur_metrics.edge_density
        })

    # Save to Excel
    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)

    print(f"Results saved to {output_excel}")

    print("Generating 3D scatter plot ...")
    plot_3d_scatter(df, output_image)


def plot_3d_scatter(df, output_image):
    # Extract the required columns
    cci = df['Color Cast Index']  # Color Cast Index
    dpp = df['Dark Pixel Percentage (%)']  # Dark Pixel Percentage
    blur = df['Blur Amount']  # Blur Amount

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the scatter points with color mapping based on the blur amount
    scatter = ax.scatter(cci, dpp, blur, c=blur, cmap='viridis', s=50, alpha=0.8)

    # Set image resolution, 6000x4000 pixels
    fig.set_size_inches(20, 12)

    # Add a color bar indicating the blur amount
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Blur Amount')
    plt.rcParams.update({'font.size': 24})

    # Set labels for the axes
    ax.set_xlabel('Color Cast Index', fontsize=24)
    ax.set_ylabel('Dark Pixel Percentage (%)', fontsize=24)
    ax.set_zlabel('Blur Amount', fontsize=24)

    # Set the size of tick labels
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Set the title of the plot
    ax.set_title('3D Scatter Plot of Underwater Image Metrics', fontsize=16)

    # Adjust the viewing angle
    ax.view_init(elev=20, azim=45)  # Adjust elevation and azimuth as needed

    # Display grid lines
    ax.grid(True)

    # Show the plot
    # plt.show()

    # Save the plot as an image
    fig.savefig(output_image, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    process_images(DATASET_FOLDER, OUTPUT_EXCEL, OUTPUT_IMAGE)
