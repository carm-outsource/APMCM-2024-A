# processing.py - Updated Processing with Multi-Color Space Fusion
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from model import UWCNN_Enhanced


def rgb_to_lab(image):
    return image.convert('LAB')

def rgb_to_hsv(image):
    return image.convert('HSV')

def process_images(model_path, input_dir, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)

    model = UWCNN_Enhanced()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
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

                # Use only RGB channels as the model expects 3 channels
                img_rgb = transform(img).unsqueeze(0).to(device)

                output = model(img_rgb)
                output = output.squeeze(0).cpu()
                output_image = transforms.ToPILImage()(output.clamp(0, 1))

                output_image.save(os.path.join(output_dir, img_name))
            else:
                print(f"Skipping non-image file: {img_name}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = 'models/uwcnn_enhanced.pth'
    input_dir = 'data/test/input/'
    output_dir = 'output/enhanced/'

    process_images(model_path, input_dir, output_dir, device)
