import os
import torch
from torchvision import transforms
from PIL import Image
from model import CombinedNet
from tqdm import tqdm

def process_image(image_path, model, device):
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Apply transform
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output_tensor = model(image_tensor)
        output_tensor = output_tensor.squeeze(0).cpu()

    # Convert output tensor to image
    output_image = transforms.ToPILImage()(output_tensor.clamp(0, 1))

    # Resize back to original size if desired
    output_image = output_image.resize(original_size, Image.BICUBIC)

    return output_image

if __name__ == '__main__':
    model_path = 'models/combined_net.pth'
    input_dir = 'data/test/input/'
    output_dir = 'output/v2/'
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CombinedNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('jpg','jpeg','png','bmp','tiff','tif'))]

    for image_name in tqdm(image_files):
        input_path = os.path.join(input_dir, image_name)
        output_image = process_image(input_path, model, device)
        output_path = os.path.join(output_dir, image_name)
        output_image.save(output_path)
