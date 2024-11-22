# training.py - Updated Training Script with Advanced Loss Functions
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image
from model import UWCNN_Enhanced
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision.models import VGG16_Weights
import numpy as np


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
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
    # Get max height and width in this batch
    max_height = max([img.shape[1] for img in inputs])
    max_width = max([img.shape[2] for img in inputs])

    padded_inputs = []
    padded_targets = []

    for img_in, img_tgt in zip(inputs, targets):
        # Calculate padding sizes
        pad_width = max_width - img_in.shape[2]
        pad_height = max_height - img_in.shape[1]
        # Padding format: (left, top, right, bottom)
        padding = (0, 0, pad_width, pad_height)
        pad_in = TF.pad(img_in, padding)
        pad_tgt = TF.pad(img_tgt, padding)
        padded_inputs.append(pad_in)
        padded_targets.append(pad_tgt)

    inputs_tensor = torch.stack(padded_inputs)
    targets_tensor = torch.stack(padded_targets)
    return inputs_tensor, targets_tensor


def calculate_uiqm(image_tensor):
    """Calculate UIQM for an image represented as a tensor."""
    image_np = image_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255.0
    image_np = image_np.astype(np.uint8)

    # Convert RGB to individual channels
    R = image_np[:, :, 0].astype(np.float32)
    G = image_np[:, :, 1].astype(np.float32)
    B = image_np[:, :, 2].astype(np.float32)

    # Calculate UICM
    RG = R - G
    YB = 0.5 * (R + G) - B
    uicm = np.mean(np.abs(RG)) + np.mean(np.abs(YB))

    # Calculate UISM
    gradient_R = np.gradient(R)
    gradient_G = np.gradient(G)
    gradient_B = np.gradient(B)
    gradient_magnitude = np.sqrt(
        gradient_R[0] ** 2 + gradient_R[1] ** 2 + gradient_G[0] ** 2 + gradient_G[1] ** 2 + gradient_B[0] ** 2 +
        gradient_B[1] ** 2)
    uism = np.mean(gradient_magnitude)

    # Calculate UIConM
    std_R = np.std(R)
    std_G = np.std(G)
    std_B = np.std(B)
    uiconm = np.mean([std_R, std_G, std_B])

    # UIQM calculation
    c1 = 0.0282
    c2 = 0.2953
    c3 = 3.5753
    uiqm = c1 * uicm + c2 * uism + c3 * uiconm
    return uiqm


def uiqm_criterion(output, target):
    output_uiqm = calculate_uiqm(output[0].detach())  # Assuming batch size of 1 for simplicity
    target_uiqm = calculate_uiqm(target[0].detach())
    loss = torch.abs(torch.tensor(output_uiqm) - torch.tensor(target_uiqm))
    return loss


def train_model(model, dataloader, criterion, perceptual_criterion, optimizer, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(dataloader, total=len(dataloader))
        for inputs, targets in loop:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            # Compute different losses
            loss_pixel = criterion(outputs, targets)
            loss_perceptual = perceptual_criterion(outputs, targets)
            loss_uiqm = uiqm_criterion(outputs, targets)

            # Total loss with weighted sum
            loss = loss_pixel + 0.1 * loss_perceptual + 0.01 * loss_uiqm
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(loss=running_loss / (loop.n + 1))
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
    return model


if __name__ == '__main__':
    num_epochs = 50
    batch_size = 4  # Adjust batch size as needed
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_input_dir = 'data/train/input/'
    train_target_dir = 'data/train/target/'

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = UnderwaterDataset(train_input_dir, train_target_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    model = UWCNN_Enhanced()
    criterion = nn.L1Loss()
    perceptual_criterion = PerceptualLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model = train_model(model, train_loader, criterion, perceptual_criterion, optimizer, num_epochs, device)

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/uwcnn_enhanced.pth')