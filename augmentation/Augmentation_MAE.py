import torch

def random_bbox(size, pad_size):
    C, H, W = size

    cx = torch.randint(0, W, (1,)).item()
    cy = torch.randint(0, H, (1,)).item()

    bbx1 = max(0, cx - pad_size // 2)
    bby1 = max(0, cy - pad_size // 2)
    bbx2 = min(W, cx + pad_size // 2)
    bby2 = min(H, cy + pad_size // 2)

    return bbx1, bby1, bbx2, bby2

def random_cutout(img, pad_size, mask_ratio, replace=0):
    _, h, w = img.shape
    total_area = h * w
    box_area = pad_size * pad_size

    # Kaç kutu gerektiğini hesapla (örneğin %50 maskelenmek istiyorsa)
    num_boxes = int((mask_ratio * total_area) / box_area)

    cutout_img = img.clone()
    mask_map = torch.zeros_like(img)

    for _ in range(num_boxes):
        bbx1, bby1, bbx2, bby2 = random_bbox(img.shape, pad_size)
        cutout_img[:, bbx1:bbx2, bby1:bby2] = replace
        mask_map[:, bbx1:bbx2, bby1:bby2] = 1

    return cutout_img, mask_map

def Cutout(images, pr, pad_size, mask_ratio):
    replace = 0
    B, C, H, W = images.shape

    cutout_images = []
    mask_maps = []

    for i in range(B):
        if torch.rand(1) < pr:
            cutout_image, mask_map = random_cutout(images[i], pad_size, mask_ratio, replace)
        else:
            cutout_image = images[i].clone()
            mask_map = torch.zeros_like(images[i])

        cutout_images.append(cutout_image)
        mask_maps.append(mask_map)

    images = torch.stack(cutout_images, dim=0)
    masks = torch.stack(mask_maps, dim=0)

    return images, masks
