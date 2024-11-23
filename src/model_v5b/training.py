import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image
from model import ImprovedUWCNN
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import numpy as np

def calculate_uiqm(image_tensor):
    """
    Calculate the UIQM (Underwater Image Quality Measure) for a given image.
    Args:
        image_tensor (torch.Tensor): Input image tensor with shape (B, C, H, W) or (C, H, W), values in range [0, 1].
    Returns:
        float: UIQM value.
    """
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor[0]  # Use the first image in the batch if batch dimension is present
    image_tensor = torch.clamp(image_tensor, 0, 1)  # Ensure values are in [0, 1]
    image = image_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255  # Convert to HWC format, detach, and scale to [0, 255]
    image = image.astype(np.float32)

    # UICM (Colorfulness measure)
    rg = image[:, :, 0] - image[:, :, 1]
    yb = 0.5 * (image[:, :, 0] + image[:, :, 1]) - image[:, :, 2]
    uicm = np.sqrt(np.mean(rg ** 2) + np.mean(yb ** 2))

    # UISM (Sharpness measure)
    uism = np.mean(np.abs(np.gradient(image[:, :, 0])) + np.abs(np.gradient(image[:, :, 1])) + np.abs(np.gradient(image[:, :, 2])))

    # UIConM (Contrast measure)
    ui_contrast = image.max() - image.min()
    uiconm = ui_contrast / 255.0

    # Combining the three measures
    uiqm_value = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
    return uiqm_value


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        from torchvision.models import VGG16_Weights
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        loss = F.l1_loss(x_vgg, y_vgg)
        return loss


class UnderwaterDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.input_images = sorted(os.listdir(input_dir))
        self.target_images = sorted(os.listdir(target_dir))

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image = Image.open(os.path.join(self.input_dir, self.input_images[idx])).convert('RGB')
        target_image = Image.open(os.path.join(self.target_dir, self.target_images[idx])).convert('RGB')

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image


def collate_fn(batch):
    inputs, targets = zip(*batch)
    max_height = max([img.shape[1] for img in inputs])
    max_width = max([img.shape[2] for img in inputs])

    padded_inputs = []
    padded_targets = []

    for img_in, img_tgt in zip(inputs, targets):
        pad_width = max_width - img_in.shape[2]
        pad_height = max_height - img_in.shape[1]
        padding = (0, 0, pad_width, pad_height)
        pad_in = TF.pad(img_in, padding)
        pad_tgt = TF.pad(img_tgt, padding)
        padded_inputs.append(pad_in)
        padded_targets.append(pad_tgt)

    inputs_tensor = torch.stack(padded_inputs)
    targets_tensor = torch.stack(padded_targets)
    return inputs_tensor, targets_tensor


def train_model(model, dataloader, criterion, perceptual_criterion, optimizer, num_epochs, device):
    model = nn.DataParallel(model)  # Wrap the model with DataParallel to use multiple GPUs
    model.to(device)
    scaler = torch.amp.GradScaler()  # Use mixed precision training to reduce memory usage
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(dataloader, total=len(dataloader))
        for inputs, targets in loop:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            try:
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(inputs)
                    outputs = torch.clamp(outputs, 0, 1)
                    loss_pixel = criterion(outputs, targets)
                    loss_perceptual = perceptual_criterion(outputs, targets)
                    loss_uiqm = -calculate_uiqm(outputs)  # Minimize negative UIQM to maximize quality
                    loss = loss_pixel + 0.1 * loss_perceptual + 0.01 * loss_uiqm
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print("Out of memory error caught, skipping batch")
                    torch.cuda.empty_cache()
                else:
                    raise e
            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(loss=running_loss / (loop.n + 1))
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
    return model


if __name__ == '__main__':
    num_epochs = 100
    batch_size = 4
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_input_dir = 'data/train/input/'
    train_target_dir = 'data/train/target/'

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = UnderwaterDataset(train_input_dir, train_target_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)

    model = ImprovedUWCNN()
    criterion = nn.L1Loss()
    perceptual_criterion = PerceptualLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model = train_model(model, train_loader, criterion, perceptual_criterion, optimizer, num_epochs, device)

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/uwcnn_v5b.pth')
