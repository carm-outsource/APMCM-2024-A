import os
import cv2
import numpy as np
from scipy import stats
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import pandas as pd
from tqdm import tqdm
from skimage import color

SOURCE_FOLDER = 'data/test/input'
TARGET_FOLDER = 'output/v5'
RESULT_OUTPUT = 'output/image_quality_metrics-v5.csv'


def calculate_uciqe(image):
    img_lab = color.rgb2lab(image)
    L, A, B = img_lab[:, :, 0], img_lab[:, :, 1], img_lab[:, :, 2]
    chroma = np.sqrt(A ** 2 + B ** 2)

    # Mean of chroma
    c_mean = np.mean(chroma)

    # Standard deviation of chroma
    c_std = np.std(chroma)

    # Contrast of L channel
    l_con = np.std(L)

    # UCIQE formula
    uciqe = 0.4680 * c_std + 0.2745 * c_mean + 0.2576 * l_con
    return uciqe


# Function to calculate UIQM
# UIQM calculation requires three components: UICM, UISM, and UIConM
def calculate_uicm(image):
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    uicm = np.std(rg) + np.std(yb) - 0.026 * np.mean(rg) - 0.1586 * np.mean(yb)
    return uicm


def calculate_uism(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    uism = np.mean(np.abs(laplacian))
    return uism


def calculate_uiconm(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    con_std = np.std(gray)
    return con_std


def calculate_uiqm(image):
    uicm = calculate_uicm(image)
    uism = calculate_uism(image)
    uiconm = calculate_uiconm(image)
    uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
    return uiqm


# Function to calculate PSNR between two images
def calculate_psnr(image1, image2):
    return compare_psnr(image1, image2)


# Function to recursively get all image paths in a folder
def get_all_image_paths(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths


# Function to process all images and calculate metrics
def process_images(input_folder, processed_folder, output_csv):
    input_image_paths = get_all_image_paths(input_folder)
    processed_image_paths = get_all_image_paths(processed_folder)
    results = []

    for img_path in tqdm(input_image_paths, desc="Processing images"):
        # Read input and processed images
        try:
            image_name = os.path.basename(img_path)
            processed_img_path = os.path.join(processed_folder, image_name)
            if not os.path.exists(processed_img_path):
                print(f"Processed image not found for {img_path}")
                continue

            input_image = io.imread(img_path)
            processed_image = io.imread(processed_img_path)

            if input_image.shape[-1] == 4:
                input_image = input_image[:, :, :3]  # Remove alpha channel if present
            if processed_image.shape[-1] == 4:
                processed_image = processed_image[:, :, :3]  # Remove alpha channel if present
        except Exception as e:
            print(f"Could not read image {img_path} or {processed_img_path}: {e}")
            continue

        # Calculate metrics
        psnr = calculate_psnr(input_image, processed_image)
        uciqe_source = calculate_uciqe(input_image)
        uiqm_source = calculate_uiqm(input_image)
        uciqe_processed = calculate_uciqe(processed_image)
        uiqm_processed = calculate_uiqm(processed_image)

        # Save results
        results.append({
            'Image': img_path,
            'PSNR': psnr,
            'UCIQE-Source': uciqe_source,
            'UCIQE-Processed': uciqe_processed,
            'UIQM-Source': uiqm_source,
            'UIQM-Processed': uiqm_processed
        })

    # Save results to a CSV file
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


if __name__ == '__main__':
    process_images(SOURCE_FOLDER, TARGET_FOLDER, RESULT_OUTPUT)
