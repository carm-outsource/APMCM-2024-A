import os
import torch
from PIL import Image, ImageEnhance
from torchvision import transforms
from model import ImprovedUWCNN
from tqdm import tqdm
import numpy as np


def auto_white_balance(image):
    """
    Automatically adjust the white balance of the image.
    Args:
        image (PIL.Image): Input image in RGB format.
    Returns:
        PIL.Image: White-balanced image.
    """
    # Convert to numpy array
    img_np = np.array(image)

    # Compute the average of each channel
    r_avg, g_avg, b_avg = np.mean(img_np[:,:,0]), np.mean(img_np[:,:,1]), np.mean(img_np[:,:,2])

    # Compute the scaling factors
    avg = (r_avg + g_avg + b_avg) / 3
    r_gain = avg / r_avg
    g_gain = avg / g_avg
    b_gain = avg / b_avg

    # Apply the gains to each channel
    img_np[:,:,0] = np.clip(img_np[:,:,0] * r_gain, 0, 255)
    img_np[:,:,1] = np.clip(img_np[:,:,1] * g_gain, 0, 255)
    img_np[:,:,2] = np.clip(img_np[:,:,2] * b_gain, 0, 255)

    # Convert back to PIL Image
    return Image.fromarray(np.uint8(img_np))


def dehaze(image):
    """
    Simulate the 'dehaze' effect by enhancing contrast in the image.
    Args:
        image (PIL.Image): Input image in RGB format.
    Returns:
        PIL.Image: Dehazed image.
    """
    enhancer = ImageEnhance.Contrast(image)
    image_enhanced = enhancer.enhance(1.5)  # Increase contrast to simulate dehazing
    return image_enhanced


def clarity(image):
    """
    Enhance the local contrast to simulate clarity effect (enhance fine details).
    Args:
        image (PIL.Image): Input image in RGB format.
    Returns:
        PIL.Image: Image with enhanced clarity.
    """
    enhancer = ImageEnhance.Sharpness(image)
    image_enhanced = enhancer.enhance(2.0)  # Sharpen the image to enhance clarity
    return image_enhanced


def process_images(model_path, input_dir, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)

    # Load the V5 model (Assuming model is named 'ImprovedUWCNN_v5')
    model = ImprovedUWCNN()
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()

    # Use DataParallel to handle multiple GPUs
    model = torch.nn.DataParallel(model)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    input_images = sorted(os.listdir(input_dir))

    with torch.no_grad():
        for img_name in tqdm(input_images, desc='Processing Images'):
            img_path = os.path.join(input_dir, img_name)
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                img = Image.open(img_path).convert('RGB')

                # Apply automatic adjustments
                # img = auto_white_balance(img)  # Apply automatic white balance
                img = dehaze(img)  # Apply dehaze effect
                img = clarity(img)  # Enhance clarity

                # Transform image for model processing
                img_input = transform(img).unsqueeze(0).to(device)

                # Pass through V5 model
                output = model(img_input)
                output = torch.clamp(output, 0, 1)
                output_image = transforms.ToPILImage()(output.squeeze(0).cpu())

                # Save the processed output image
                output_image.save(os.path.join(output_dir, img_name))
            else:
                print(f"Skipping non-image file: {img_name}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = 'models/uwcnn_v5b.pth'  # Ensure this path points to your v5 model
    input_dir = 'data/test/input/'
    output_dir = 'output/v5b/'  # Updated output folder to 'v5s'

    process_images(model_path, input_dir, output_dir, device)