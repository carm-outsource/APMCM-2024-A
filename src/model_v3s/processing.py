import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from model import UWCNN


def process_images(model_path, input_dir, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)

    model = UWCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
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
                img_input = transform(img).unsqueeze(0).to(device)

                output = model(img_input)
                output = output.squeeze(0).cpu()
                output_image = transforms.ToPILImage()(output.clamp(0, 1))

                output_image.save(os.path.join(output_dir, img_name))
            else:
                print(f"Skipping non-image file: {img_name}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = 'models/uwcnn_v3s.pth'
    input_dir = 'data/test/input/'
    output_dir = 'output/v3s/'

    process_images(model_path, input_dir, output_dir, device)
