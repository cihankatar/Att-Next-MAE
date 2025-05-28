import torch
import torch.nn as nn
#import time
#from models.Metaformer import caformer_s18_in21ft1k
from models.enc import encoder_function
from models.dec import decoder_function

def device_f():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv      = nn.Conv2d(64,64,3,padding='same',groups=64)
        self.pwconv1   = nn.Linear(64,64)
        self.norm      = nn.LayerNorm(64)
        self.act       = nn.GELU()
        self.pwconv2   = nn.Linear(64,3)
    def forward(self,out):
        out = self.conv(out)
        out = self.norm(out.permute(0, 2, 3, 1))
        out = self.act(out)

        out = self.pwconv1(out)
        out = self.norm(out)
        out = self.act(out)

        out = self.pwconv2(out)
        out = out.permute(0, 3, 1, 2)
        return out 

class Bottleneck(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        med_channels    = int(4 * in_c)

        self.dwconv1     = nn.Conv2d(in_c, in_c, kernel_size=3, padding="same", groups=in_c)
        self.pwconv1    = nn.Linear(in_c, med_channels)
        self.pwconv2    = nn.Linear(med_channels, out_c)
        self.norm       = nn.LayerNorm(out_c)    
        self.act        = nn.GELU()

        #self.attention = Attention(dim=512)
               
    def forward(self, inputs):  
        #x = inputs.permute(0, 2, 3, 1)
        #x   =   x + self.norm(self.attention(inputs.permute(0, 2, 3, 1)))
        x   =   self.dwconv1(inputs).permute(0, 2, 3, 1)
        x   =   self.norm(x)     
        x   =   self.pwconv1(x)
        x   =   self.act(x)
        convout = self.pwconv2(x).permute(0, 3, 1, 2)
        out     = convout+inputs

        return out
    
#####   MODEL #####
    
class model_dice_bce(nn.Module):
    def __init__(self,training_mode):
        super().__init__()
        
        self.training_mode = training_mode
        self.bottleneck    = Bottleneck(512, 512)
        size_dec           = [512,256,128,64]
        
        self.encoder       = encoder_function()

        if self.training_mode == "ssl_pretrained":
            for param in self.encoder.parameters():
                param.requires_grad = False  

        self.decoder       = decoder_function()
        self.head          = Head()

    def forward(self, inputs):                      # 1x  3 x 128 x 128
        
        # ENCODER   
        if self.training_mode ==  "ssl_pretrained":
            en_features = inputs
        else:
            en_features = self.encoder(inputs)               # [2, 64, 64, 64]) ([2, 128, 32, 32]) [2, 320, 16, 16]) ([2, 512, 8, 8])
        
        skip_connections = en_features[:3][::-1]

        b   = self.bottleneck(en_features[3])                              # 1x 512 x 8x8

        out = self.decoder(b,skip_connections) 

        out = self.head(out)

        return out


if __name__ == "__main__":

    #start=time.time()

    x = torch.randn((2, 3, 256, 256))
    #f = CA_CBA_Convnext(1)
    #y = f(x)
    #print(x.shape)
    #print(y.shape)

    #end=time.time()
    
    #print(f'spending time :  {end-start}')








