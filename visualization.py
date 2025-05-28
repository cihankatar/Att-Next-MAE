
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch


def MAE(i,out,images):
    import matplotlib.pyplot as plt
#   which_batch=np.random.randint(2)
#   which_image=np.random.randint(7)
    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(images[i].permute(2,1,0))
    plt.subplot(2,1,2)
    plt.imshow(out[i].mean(dim=0).detach().cpu())
