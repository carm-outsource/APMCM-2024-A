import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from model import UWCNN_Enhanced


def load_model(model_path, device):
    """Load the trained UWCNN_Enhanced model."""
    model = UWCNN_Enhanced().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def process_image(model, image_path, output_path, device):
    """Process a single underwater image using the trained model."""
    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Run the model to enhance the image
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Convert the output tensor to an image
    output_image = transforms.ToPILImage()(output_tensor.squeeze().cpu().clamp(0, 1))
    output_image.save(output_path)


def batch_process_images(model, input_dir, output_dir, device):
    """Batch process all images in the input directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_images = sorted(os.listdir(input_dir))

    with torch.no_grad():
        for image_name in tqdm(input_images, desc='Processing Images'):
            input_path = os.path.join(input_dir, image_name)
            output_path = os.path.join(output_dir, image_name)

            if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                print(f"Processing {image_name}...")
                process_image(model, input_path, output_path, device)
                print(f"Saved enhanced image to {output_path}")
            else:
                print(f"Skipping non-image file: {image_name}")


if __name__ == '__main__':
    # Set the device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths for the model and data
    model_path = 'models/uwcnn_v4.pth'  # Path to the trained model
    input_dir = 'data/test/input/'  # Directory containing images to be enhanced
    output_dir = 'data/test/output/'  # Directory to save enhanced images

    # Load the trained model
    model = load_model(model_path, device)

    # Batch process the images
    batch_process_images(model, input_dir, output_dir, device)