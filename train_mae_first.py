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
    return clean.clamp(0, 1)


def masked_mse_loss(reconstructed, original, mask_map):
    # mask_map: 1 = maskelenmiş (tahmin edilmesi gereken), 0 = görünür
    masked_pred = reconstructed * mask_map
    masked_gt   = original * mask_map
    loss = F.mse_loss(masked_pred, masked_gt, reduction='sum')
    norm = mask_map.sum() * reconstructed.shape[1]  # B x C x H x W
    return loss / (norm + 1e-6)

def sobel_edge_map(img):
    # img: [B, 3, H, W] → convert to grayscale
    gray = img.mean(dim=1, keepdim=True)  # [B, 1, H, W]
    
    # Define Sobel filters
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)

    edge_x = F.conv2d(gray, sobel_x, padding=1)
    edge_y = F.conv2d(gray, sobel_y, padding=1)
    edge = torch.sqrt(edge_x**2 + edge_y**2)
    
    # edge = edge / (edge.max() + 1e-6)  # normalize
    return edge  # [B, 1, H, W]


def log_edge_map(img):
    gray = img.mean(dim=1, keepdim=True)  # [B, 1, H, W]

    # Laplacian of Gaussian kernel (5x5)
    log_kernel = torch.tensor([
        [0,  0, -1,  0,  0],
        [0, -1, -2, -1,  0],
        [-1,-2, 16, -2, -1],
        [0, -1, -2, -1,  0],
        [0,  0, -1,  0,  0]
    ], dtype=torch.float32, device=img.device).view(1, 1, 5, 5)

    edge = F.conv2d(gray, log_kernel, padding=2)
    edge = torch.abs(edge)
    return edge

import math

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

def mae_loss(x_rec, x_orig,mask_map):

    l1                      = masked_mse_loss(x_rec,x_orig,mask_map)
    
    edge_rec_gabor          = gabor(x_rec)
    edge_removehair_s_gabor = gabor(x_orig)
    edge_l                  = masked_mse_loss(edge_rec_gabor, edge_removehair_s_gabor, mask_map)

    #edge_rec            = sobel_edge_map(x_rec)
    #edge_removehair_s   = sobel_edge_map(x_orig)
    #edge_l              = masked_mse_loss(edge_rec, edge_removehair_s, mask_map)

    return l1, edge_l   # you can weight this

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

@torch.no_grad()
def update_teacher(student, teacher, momentum):
    for param_s, param_t in zip(student.parameters(), teacher.parameters()):
        param_t.data = momentum * param_t.data + (1. - momentum) * param_s.data

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

    model       = model_dice_bce(args.mode).to(device)
    head        = model.head
    encoder     = model.encoder
    bottleneck  = model.bottleneck
    decoder     = model.decoder

    # Optimizasyon & Loss
    checkpoint_path = folder_path+str(encoder.__class__.__name__)+str(res)
    optimizer       = AdamW(model.parameters(), lr=config['learningrate'],weight_decay=0.05)
    scheduler       = CosineAnnealingLR(optimizer, config['epochs'], eta_min=config['learningrate'] / 10)
    
    print(f"Training on {len(train_loader) * args.bsize} images. Saving checkpoints to {folder_path}")
    print('Train loader transform',train_loader.dataset.tr)
    print('Val loader transform',val_loader.dataset.tr)
    print(f"model config : {checkpoint_path}")

    # ML_DATA_OUTPUT      = os.environ["ML_DATA_OUTPUT"]+'isic_1/'
    # checkpoint_path_read = ML_DATA_OUTPUT+str(encoder.__class__.__name__)+str(res)
    # encoder.load_state_dict(torch.load(checkpoint_path_read, map_location=torch.device('cpu')))
    
    gabor_weight = 1

    def run_epoch(loader,epoch_idx,training=True):
        """Run a single training or validation epoch."""
        epoch_loss,epoch_p_loss,epoch_edge_loss  = 0.0,0.0,0.0
        model.train()
 # or whatever tested value

        current_lr = optimizer.param_groups[0]['lr']
        print("current_lr",current_lr,"\n")

        with torch.set_grad_enabled(training):
            for batch_idx, images in enumerate(loader):  #tqdm(loader, desc="Training" if training else "Validating", leave=False):
                
                # 1. Random masking (cutout gibi) + mask haritası oluştur
                removehair    = dullrazor (images)
                masked_images, mask_map = Cutout(removehair, pr=1, pad_size=10, mask_ratio=0.5)
                removehair        = removehair.to(device)
                masked_images = masked_images.to(device)
                mask_map      = mask_map.to(device)

                # 2. Model tahmin etsin
                features         = encoder(masked_images)
                out              = bottleneck(features[3])
                skip_connections = features[:3][::-1]
                reconstructed    = decoder(out,skip_connections)
                out              = head(reconstructed)

                p_loss, edge_loss = mae_loss(out, removehair, mask_map)
                loss = p_loss + gabor_weight * edge_loss                
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                epoch_loss      += loss.item() 
                epoch_p_loss    += p_loss.item() 
                epoch_edge_loss += edge_loss.item() 

                log_images = (not training and batch_idx == 0)

                # Only log 1 image during validation per epoch
                if log_images:
                    original_img        = images[0].detach().cpu()
                    remove_hair         = removehair[0].detach().cpu()
                    masked_img          = masked_images[0].detach().cpu()
                    reconstructed_img   = out[0].detach().cpu()

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

                    img_grid = [
                        wandb.Image(to_img(original_img), caption="Original"),
                        wandb.Image(to_img(remove_hair), caption="Hair Removed Input"),
                        wandb.Image(to_img(masked_img), caption="Masked Input"),
                        wandb.Image(to_img(reconstructed_img), caption="Reconstructed"),
                        wandb.Image(to_edge_img(removehair), caption="Gabor Mask Input"),
                        wandb.Image(to_edge_img(out), caption="Gabor Mask Output")
                    ]

                    wandb.log({f"Reconstruction Epoch {epoch_idx}": img_grid})

        return epoch_loss / len(loader), epoch_p_loss/len(loader) , epoch_edge_loss/len(loader)

    epoch_idx=0
    for epoch in trange(config['epochs'], desc="Epochs"):

        # Training
        train_loss, epoch_p_loss , epoch_edge_loss = run_epoch(train_loader, epoch_idx,training=True )
        wandb.log({"Train Loss": train_loss})
        wandb.log({"Train p_Loss": epoch_p_loss })
        wandb.log({"Train edge_Loss": epoch_edge_loss })
        scheduler.step()

        val_loss, v_epoch_p_loss , v_epoch_edge_loss = run_epoch(val_loader, epoch_idx,training=False)
        wandb.log({"Validation Loss": val_loss })
        wandb.log({"Validation p_Loss": v_epoch_p_loss })
        wandb.log({"Validation edge_Loss": v_epoch_edge_loss })

        epoch_idx+=1

        print("epoch_idx",epoch_idx,"\n")
        print(f"Train Loss: {train_loss:.6f}, p_Loss: {epoch_p_loss:.6f},edge_Loss: {epoch_edge_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}, v_p_loss: {v_epoch_p_loss:.6f},v_edge_loss: {v_epoch_edge_loss:.6f}")

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(encoder.state_dict(), checkpoint_path)
            print(f"Best model saved")

    wandb.finish()

if __name__ == "__main__":
    main()
