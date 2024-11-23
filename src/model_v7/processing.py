import os
import torch
from PIL import Image, ImageEnhance
from torchvision import transforms
from model import ImprovedUWCNN
from tqdm import tqdm


def adjust_white_balance(image):
    """
    Adjust white balance of the given image by enhancing its color balance.
    Args:
        image (PIL.Image): Input image in RGB format.
    Returns:
        PIL.Image: White-balanced image.
    """
    enhancer = ImageEnhance.Color(image)
    image_enhanced = enhancer.enhance(1.2)  # Reduced enhancement to avoid over-brightening
    return image_enhanced


def adjust_color_saturation(image):
    """
    Adjust the color saturation of the given image to enhance vibrancy.
    Args:
        image (PIL.Image): Input image in RGB format.
    Returns:
        PIL.Image: Color saturation enhanced image.
    """
    enhancer = ImageEnhance.Color(image)
    image_enhanced = enhancer.enhance(1.8)  # Reduced saturation enhancement
    return image_enhanced


def adjust_contrast(image):
    """
    Adjust the contrast of the given image to improve depth and definition.
    Args:
        image (PIL.Image): Input image in RGB format.
    Returns:
        PIL.Image: Contrast-enhanced image.
    """
    enhancer = ImageEnhance.Contrast(image)
    image_enhanced = enhancer.enhance(1.5)  # Add contrast to improve image depth
    return image_enhanced


def process_images(model_path, input_dir, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)

    model = ImprovedUWCNN()
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    input_images = sorted(os.listdir(input_dir))

    with torch.no_grad():
        for img_name in tqdm(input_images, desc='Processing Images'):
            img_path = os.path.join(input_dir, img_name)
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                img = Image.open(img_path).convert('RGB')

                img = adjust_white_balance(img)
                img = adjust_color_saturation(img)
                img = adjust_contrast(img)  # Add contrast adjustment

                img_input = transform(img).unsqueeze(0).to(device)

                output = model(img_input)
                output = torch.clamp(output, 0, 1)
                output_image = transforms.ToPILImage()(output.squeeze(0).cpu())

                output_image.save(os.path.join(output_dir, img_name))
            else:
                print(f"Skipping non-image file: {img_name}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = 'models/uwcnn_v7.pth'
    input_dir = 'data/test/input/'
    output_dir = 'output/v7/'

    process_images(model_path, input_dir, output_dir, device)