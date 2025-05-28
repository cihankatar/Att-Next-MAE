##IMPORT 
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class dataset(Dataset):
    def __init__(self,train_path,mask_path,cutout_pr,cutout_box,transforms,training_type): #
        super().__init__()
        self.train_path      = train_path
        self.mask_path       = mask_path
        self.tr       = transforms
        self.cutout_pr=cutout_pr
        self.cutout_pad=cutout_box
        self.training_type = training_type

    def __len__(self):
         return len(self.train_path)
    
    def __getitem__(self,index):        

            image = Image.open(self.train_path[index])
            image = np.array(image,dtype=float)
            image = image.astype(np.float32)
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)

            image=image/255
            image=self.tr(image)

            return image 
    
    