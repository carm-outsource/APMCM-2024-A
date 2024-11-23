import os
import torch
from PIL import Image
from torchvision import transforms
from model import ImprovedUWCNN
from tqdm import tqdm

def process_images(model_path, input_dir, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)

    # Load the model
    model = ImprovedUWCNN()
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

                # Process the image with the model
                output = model(img_input)
                output = output.squeeze(0).cpu()
                output_image = transforms.ToPILImage()(output.clamp(0, 1))

                # Save the processed image
                output_image.save(os.path.join(output_dir, img_name))
            else:
                print(f"Skipping non-image file: {img_name}")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = 'models/uwcnn_v5.pth'
    input_dir = 'data/test/input/'
    output_dir = 'output/v5/'

    process_images(model_path, input_dir, output_dir, device)