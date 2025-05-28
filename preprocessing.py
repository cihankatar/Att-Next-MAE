import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

# Directories
input_dir = "/Users/input/data/ckatar/isic_2018_1/train/images"
output_dir = "/Users/input/data/ckatar/isic_2018_1/train/reconstruction_003"
os.makedirs(output_dir, exist_ok=True)

# Cosine similarity threshold
SIM_THRESHOLD = 1.0

# Transformations
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()



def detect_dark_border(gray_tensor, threshold=0.1, border_width=20):
    """
    Detect dark regions near the border of the image.
    Args:
        gray_tensor: [B, 1, H, W] tensor with values in [0, 1]
    Returns:
        mask: [B, 1, H, W] binary mask of dark border regions (float)
    """
    B, _, H, W = gray_tensor.shape
    mask = torch.zeros_like(gray_tensor, dtype=torch.bool)

    # Detect boolean masks for each border
    top    = gray_tensor[:, :, :border_width, :] < threshold
    bottom = gray_tensor[:, :, -border_width:, :] < threshold
    left   = gray_tensor[:, :, :, :border_width] < threshold
    right  = gray_tensor[:, :, :, -border_width:] < threshold

    # Combine them into a single boolean mask
    mask[:, :, :border_width, :]  = top
    mask[:, :, -border_width:, :] = bottom
    mask[:, :, :, :border_width]  |= left
    mask[:, :, :, -border_width:] |= right

    return mask.float()  # convert to float for later use


def border_fill(image, kernel_size=25, threshold=0.2, border_width=20):
    """
    Fill dark border artifacts using local smoothing.
    Args:
        image: [B, 3, H, W] input image
    """
    gray = 0.2989 * image[:, 0:1] + 0.5870 * image[:, 1:2] + 0.1140 * image[:, 2:3]
    border_mask = detect_dark_border(gray, threshold=threshold, border_width=border_width)
    border_mask_3ch = border_mask.repeat(1, 3, 1, 1)

    # Inpaint with patch_fill logic
    padding = kernel_size // 2
    ones = torch.ones_like(image)
    masked_input = image * (1 - border_mask_3ch)
    norm = F.avg_pool2d(ones * (1 - border_mask_3ch), kernel_size, 1, padding) + 1e-8
    smooth = F.avg_pool2d(masked_input, kernel_size, 1, padding) / norm

    return image * (1 - border_mask_3ch) + smooth * border_mask_3ch


# ---- DullRazor Functions ----

def blackhat_transform(gray_tensor, kernel_size=9):
    padding = kernel_size // 2
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=gray_tensor.device)
    dilated = F.max_pool2d(gray_tensor, kernel_size, 1, padding)
    closed = -F.max_pool2d(-dilated, kernel_size, 1, padding)
    blackhat = (closed - gray_tensor).clamp(0, 1)
    return blackhat

def patch_fill(image, mask, kernel_size=7):
    padding = kernel_size // 2
    ones = torch.ones_like(image)
    masked_input = image * (1 - mask)
    norm = F.avg_pool2d(ones * (1 - mask), kernel_size, 1, padding) + 1e-8
    smooth = F.avg_pool2d(masked_input, kernel_size, 1, padding) / norm
    return image * (1 - mask) + smooth * mask

def dullrazor(batch):
    B, C, H, W = batch.shape
    gray = 0.2989 * batch[:, 0] + 0.5870 * batch[:, 1] + 0.1140 * batch[:, 2]
    gray = gray.unsqueeze(1)
    blackhat = blackhat_transform(gray, kernel_size=9)
    hair_mask = (blackhat > 0.03).float()
    hair_mask_3ch = hair_mask.repeat(1, 3, 1, 1)
    clean = patch_fill(batch, hair_mask_3ch)
    return clean.clamp(0, 1), hair_mask

# Cosine similarity function
def cosine_similarity(img1, img2):
    img1_flat = img1.reshape(-1)
    img2_flat = img2.reshape(-1)
    return F.cosine_similarity(img1_flat.unsqueeze(0), img2_flat.unsqueeze(0)).item()

# ---- Main Processing Loop ----

for fname in tqdm(sorted(os.listdir(input_dir))):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(input_dir, fname)
    image_pil = Image.open(img_path).convert("RGB")
    image_tensor = to_tensor(image_pil).unsqueeze(0)  # shape: [1, 3, H, W]

    # Apply DullRazor
    processed_tensor, hair_mask = dullrazor(image_tensor)
    fill_image                  = border_fill(processed_tensor)
    # Compute cosine similarity
    similarity = cosine_similarity(image_tensor[0], processed_tensor[0])

    # Save processed if hair is found (non-zero mask) and similarity below threshold
    if torch.sum(hair_mask) > 0 and similarity < SIM_THRESHOLD:
        result_tensor = processed_tensor[0]
    else:
        result_tensor = image_tensor[0]

    # Save image with high quality
    result_img = to_pil(result_tensor.cpu())
    save_path = os.path.join(output_dir, fname)
    result_img.save(save_path, quality=95, optimize=True)
