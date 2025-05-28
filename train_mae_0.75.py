import os
import torch
import wandb 
from tqdm import tqdm, trange
from torch.optim import AdamW 
from torch.optim.lr_scheduler import CosineAnnealingLR
from data.data_loader_mae import loader
import torch.nn.functional as F
from augmentation.Augmentation_MAE import Cutout
from wandb_init import parser_init, wandb_init
from models.Model import model_dice_bce
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from matplotlib.backends.backend_pdf import PdfPages
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
import torch.nn as nn 

class SegmentationHead(nn.Module):
    def __init__(self, in_channels=512, out_channels=1, final_size=(256, 256)):
        super().__init__()
        self.final_size = final_size
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
        )

    def forward(self, x):
        x = self.head(x)  # Shape: (B, 1, 8, 8)
        x = F.interpolate(x, size=self.final_size, mode='bilinear', align_corners=False)
        return x


def rgb_to_grayscale(tensor):
    # tensor: [B, 3, H, W]
    r, g, b = tensor[:, 0:1], tensor[:, 1:2], tensor[:, 2:3]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

# Convert to (H, W, C)
def to_img(tensor):
    return tensor.permute(1, 2, 0).numpy()

# For edge images, convert single-channel to 3-channel for better visualization
def to_edge_img(img, low=2, high=98):
    edge_tensors = gabor(img)  # shape: (1, 1, H, W)
    edge_tensors = edge_tensors.repeat(1, 3, 1, 1)  # Convert to 3-channel
    edge_tensor = edge_tensors[0]  # shape: (3, H, W)

    # Normalize using percentile
    edge_np = edge_tensor.cpu().numpy()
    p_low, p_high = np.percentile(edge_np, (low, high))
    edge_np = np.clip(edge_np, p_low, p_high)
    edge_np = (edge_np - p_low) / (p_high - p_low + 1e-8)
    
    return np.transpose(edge_np, (1, 2, 0))  # (H, W, 3)

def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    return 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))

def compute_ssim_batch(pred, target):
    pred_np = pred.detach().cpu().numpy().transpose(0,2,3,1)  # BxHxWxC
    target_np = target.detach().cpu().numpy().transpose(0,2,3,1)
    return sum([ssim(p, t, channel_axis=-1, data_range=1.0) for p, t in zip(pred_np, target_np)]) / len(pred_np)

##### DULLRAZOR ###
def blackhat_transform(gray_tensor, kernel_size=9):
    padding = kernel_size // 2
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=gray_tensor.device)
    dilated = F.max_pool2d(gray_tensor, kernel_size, 1, padding)
    closed = -F.max_pool2d(-dilated, kernel_size, 1, padding)
    blackhat = (closed - gray_tensor).clamp(0, 1)
    return blackhat

def patch_fill(image, mask, kernel_size=15):
    padding = kernel_size // 2
    ones = torch.ones_like(image)
    masked_input = image * (1 - mask)
    norm = F.avg_pool2d(ones * (1 - mask), kernel_size, 1, padding) + 1e-8
    smooth = F.avg_pool2d(masked_input, kernel_size, 1, padding) / norm
    return image * (1 - mask) + smooth * mask

def dullrazor(batch,th=0.05):
    B, C, H, W = batch.shape
    gray = 0.2989 * batch[:, 0] + 0.5870 * batch[:, 1] + 0.1140 * batch[:, 2]
    gray = gray.unsqueeze(1)
    blackhat = blackhat_transform(gray, kernel_size=9)
    hair_mask = (blackhat > th).float()
    hair_mask_3ch = hair_mask.repeat(1, 3, 1, 1)
    clean = patch_fill(batch, hair_mask_3ch)
    return clean.clamp(0, 1)

##### GABOR TRASNFORM ###

def gabor_kernel(kernel_size, sigma, theta, lambd, gamma, psi, device):
    half_size = kernel_size // 2
    y, x = torch.meshgrid(torch.arange(-half_size, half_size + 1, device=device),
                          torch.arange(-half_size, half_size + 1, device=device), indexing='ij')

    x_theta = x * math.cos(theta) + y * math.sin(theta)
    y_theta = -x * math.sin(theta) + y * math.cos(theta)

    gb = torch.exp(-.5 * (x_theta**2 + gamma**2 * y_theta**2) / sigma**2)
    gb *= torch.cos(2 * math.pi * x_theta / lambd + psi)
    return gb.view(1, 1, kernel_size, kernel_size)

def gabor_edge_map(img, theta=0):
    gray = img.mean(dim=1, keepdim=True)
    kernel = gabor_kernel(kernel_size=7, sigma=2.0, theta=theta,
                          lambd=4.0, gamma=0.5, psi=0, device=img.device)
    edge = F.conv2d(gray, kernel, padding=3)
    return torch.abs(edge)

def gabor(img):
    thetas = [0, math.pi/4, math.pi/2, 3*math.pi/4]
    edges = [gabor_edge_map(img, theta) for theta in thetas]
    edges = torch.stack(edges).mean(dim=0)
    norm = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)

    # Invert so dark tumor regions become high weight
    inverted = 1.0 - norm
    return inverted


##### MASK WEIGHTED LOSS ###
def mw_mse_loss(reconstructed, clean_images, mask_map):
    # weight_map: same shape as input, defines per-pixel loss weight
    # pseudo_mask = (edge_clean_images_s_gabor > 0.35).float()  # shape: [B, 1, H, W]
    weight_map  = None
    masked_pred = reconstructed * mask_map
    masked_gt   = clean_images * mask_map
    diff        = (masked_pred - masked_gt) ** 2  # [B, C, H, W]

    if weight_map is not None:
        diff = diff * weight_map  # Apply Gabor weights

    loss = diff.sum()
    norm = mask_map.sum() * reconstructed.shape[1]  # Normalize by valid pixels
    return loss / (norm + 1e-6)


def using_device():
    """Set and print the device used for training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    return device

def setup_paths(data):
    """Set up data paths for training and validation."""
    folder_mapping = {
        "isic_2018_1": "isic_1/",
        "kvasir_1": "kvasir_1/",
        "ham_1": "ham_1/",
        "PH2Dataset": "PH2Dataset/",
        "isic_2016_1": "isic_2016_1/"
    }
    folder = folder_mapping.get(data)
    base_path = os.environ["ML_DATA_OUTPUT"] if torch.cuda.is_available() else os.environ["ML_DATA_OUTPUT_LOCAL"]
    print(base_path)
    return os.path.join(base_path, folder)


# Main Function
def main():
    # Configuration and Initial Setup
    
    data, training_mode, op = 'isic_2018_1', "ssl", "train"
    best_loss = float("inf")
    device      = using_device()
    folder_path = setup_paths(data)
    args, res   = parser_init("segmentation task", op, training_mode)
    res         = " ".join(res)
    res         = "["+res+"]"

    config      = wandb_init(os.environ["WANDB_API_KEY"], os.environ["WANDB_DIR"], args, data)

    # Data Loaders
    def create_loader(operation):
        return loader(operation,args.mode, args.sslmode_modelname, args.bsize, args.workers,
                      args.imsize, args.cutoutpr, args.cutoutbox, args.shuffle, args.sratio, data)
    
    #args.shuffle    = False
    train_loader    = create_loader(args.op)
    args.op         =  "validation"
    val_loader      = create_loader(args.op)
    args.op         = "train"

    model           = model_dice_bce(args.mode).to(device)
    head            = model.head
    encoder         = model.encoder
    bottleneck      = model.bottleneck
    decoder         = model.decoder
    segmentation_head = SegmentationHead().to(device)

    # Optimizasyon & Loss
    checkpoint_path_encoder = folder_path+str(encoder.__class__.__name__)+str(res)
    checkpoint_path_model = folder_path+str(model.__class__.__name__)+str(res)

    optimizer       = AdamW(model.parameters(), lr=config['learningrate'],weight_decay=0.05)
    scheduler       = CosineAnnealingLR(optimizer, config['epochs'], eta_min=config['learningrate'] / 10)
    print(f"Training on {len(train_loader) * args.bsize} images. Saving checkpoints to {folder_path}")
    print('Train loader transform',train_loader.dataset.tr)
    print('Val loader transform',val_loader.dataset.tr)
    print(f"model config : {checkpoint_path_encoder}")

    # ML_DATA_OUTPUT      = os.environ["ML_DATA_OUTPUT"]+'isic_1/'
    # checkpoint_path_read = ML_DATA_OUTPUT+str(encoder.__class__.__name__)+str(res)
    # encoder.load_state_dict(torch.load(checkpoint_path_read, map_location=torch.device('cpu')))
    
    def run_epoch(loader,epoch_idx,training=True):
        """Run a single training or validation epoch."""
        epoch_loss,epoch_seg_loss = 0.0,0.0
        total_ssim = 0.0
        total_psnr = 0.0
        total_samples = 0
        w=0.1
        model.train()
        current_lr = optimizer.param_groups[0]['lr']
        print("current_lr",current_lr,"\n")

        with torch.set_grad_enabled(training):
            for batch_idx, images in enumerate(loader):  #tqdm(loader, desc="Training" if training else "Validating", leave=False):
                
                clean_images            = dullrazor (images,0.06)
                pseudo_mask             = gabor(clean_images).to(device)
                masked_images, mask_map = Cutout(images, pr=1, pad_size=10, mask_ratio=0.75)
                clean_images            = clean_images.to(device)
                masked_images           = masked_images.to(device)
                mask_map                = mask_map.to(device)
                
                features         = encoder(masked_images)
                out              = bottleneck(features[3])
                skip_connections = features[:3][::-1]
                reconstructed    = decoder(out,skip_connections)
                out              = head(reconstructed)

                seg_logits = segmentation_head(features[3])
                
                # Compute segmentation loss
                seg_loss         = F.binary_cross_entropy_with_logits(seg_logits, pseudo_mask)
                seg_loss         = w*seg_loss
                loss_mse         = mw_mse_loss(out, clean_images, mask_map)
                
                loss             = loss_mse + seg_loss
              
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                epoch_loss      += loss.item() 
                epoch_seg_loss  += seg_loss.item() 

                ####  IN VALIDATION 
                if not training:
                    with torch.no_grad():
                        pred_img      = out.clamp(0, 1)
                        target_img    = clean_images.clamp(0, 1)
                        total_psnr   += compute_psnr(pred_img, target_img).item()
                        total_ssim   += compute_ssim_batch(pred_img, target_img)
                        total_samples += 1
                
                ###  LOG WANDB   # Only log 1 image during validation per epoch
                log_images = (not training and batch_idx == 0)
                if log_images:
                    original_img        = images[0].detach().cpu()
                    clean_image         = clean_images[0].detach().cpu()
                    seg_logits          = seg_logits[0].detach().cpu()
                    pseudo_mask         = pseudo_mask[0].detach().cpu()
                    masked_img          = masked_images[0].detach().cpu()
                    reconstructed_img   = out[0].detach().cpu()
                    
                    img_grid = [
                        wandb.Image(to_img(original_img),       caption="Original"),
                        wandb.Image(to_img(clean_image),        caption="Clean Image"),
                        wandb.Image(to_img(masked_img),         caption="Masked Input"),
                        wandb.Image(to_img(reconstructed_img),  caption="Predicted Image"),
                        wandb.Image(to_img(pseudo_mask),        caption="Pseudo Mask (Gabor Edges)"),
                        wandb.Image(to_img(seg_logits),         caption="Segmentation Head")

                    ]

                    wandb.log({f"Reconstruction Epoch {epoch_idx}": img_grid})
                
                        
        #### COMPUTE TOTAL LOSS AND METRIC FOR ONE EPOCH
        avg_loss = epoch_loss / len(loader)
        if not training:
            avg_psnr = total_psnr / total_samples
            avg_ssim = total_ssim / total_samples
            return avg_loss, avg_psnr, avg_ssim
        else:
            return avg_loss, epoch_seg_loss/len(loader)

    epoch_idx=0
    for epoch in trange(config['epochs'], desc="Epochs"):

        # Training
        train_loss,seg_loss = run_epoch(train_loader, epoch_idx,training=True )
        wandb.log({"Train Loss": train_loss,"Segmentation Loss:":seg_loss})

        scheduler.step()

        val_loss,avg_psnr, avg_ssim = run_epoch(val_loader, epoch_idx,training=False)
        wandb.log({"Validation Loss": val_loss,"avg_psnr": avg_psnr,"avg_ssim": avg_ssim })

        epoch_idx+=1

        print("epoch_idx",epoch_idx,"\n")
        print(f"Train Loss: {train_loss:.6f},Seg_Loss: {seg_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f},Val avg_psnr: {avg_psnr:.6f},Val avg_ssim: {avg_ssim:.6f},")

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(encoder.state_dict(), checkpoint_path_encoder)
            torch.save(model.state_dict(), checkpoint_path_model)
            print(f"Best model and encoder saved")

    wandb.finish()

if __name__ == "__main__":
    main()
