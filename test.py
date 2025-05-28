import torch
import wandb
import os 
import torchvision.transforms.functional as F
from operator import add
from tqdm import tqdm, trange
from wandb_init  import *
from visualization import *
from utils.metrics import *

from data.data_loader import loader
from augmentation.Augmentation import Cutout
#from models.MISSFormer import MISSFormer    #224
#from models.FAT_NET import FAT_Net          #224

#from models.Model4 import  model_bce_topo
#from models.Unet import UNET
#from models.DoubleUnet import build_doubleunet
#from models.Attunet import AttU_Net
#from models.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
# from models.TransUNET import TransUNet
#from models.Unet import UNET
# from models.LevitUNET import Build_LeViT_UNet_192#from models.BDFormer.BDFormer import BDFormer    #256
#from models.CFATransUnet.CFATransUnet import CFATransUnet    #224

from models.DTrAttUnet.DTrAttUnet import DTrAttUnet   #224
#from models.FCTNet.FCT_Net import FCTNet   #224
#from models.mednext.mednext2d import MedNeXtSegmentationModel   #224
#from models.RFAUCNext.rfau_cnxt import ResponseFusionAttentionUConvNextBase   #224
#from models.TransAttUnet.TransAttUnet import TransAttUnet  #256
#from models.FAT_NET import FAT_Net          #224
#from models.MISSFormer import MISSFormer    #224


def using_device():
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "") 
    return device

def process_model_output(model, images):
    """Process the output of the model based on its type."""
    model_type = model.__class__.__name__
    if model_type == 'DeepLabV3' :
        out = model(images)
        return out['out']
    
    elif model_type in ['PyramidVisionTransformerV2', 'DTrAttUnet','MedNeXtSegmentationModel']:
        out = model(images)
        return out[0]
    
    return model(images)
    
if __name__ == "__main__":

    device          = using_device()
    data            = 'isic_2018_1'
    training_mode   = "supervised"
    train           = False

    if data     == 'isic_2018_1':
        foldernamepath="isic_1/"
    elif data   == 'isic_2016_1':
        foldernamepath="isic_2016_1/"
    elif data   == 'PH2Dataset':
        foldernamepath="PH2Dataset/"
    elif data   == 'kvasir_1':
        foldernamepath="kvasir_1/"
    elif data   == 'ham_1':
        foldernamepath="ham_1/"

    WANDB_DIR           = os.environ["WANDB_DIR"]
    WANDB_API_KEY       = os.environ["WANDB_API_KEY"]
    ML_DATA_OUTPUT      = os.environ["ML_DATA_OUTPUT"]+foldernamepath

    args,res = parser_init("segmetnation_task","testing",training_mode,train)
    config              = wandb_init(WANDB_API_KEY,WANDB_DIR,args,data)

    #model = model_bce_topo(1, args.mode, args.imnetpr).to(device)
    #model = AttU_Net().to(device)

    # model1            = UNET(1).to(device)
    # model2            = build_doubleunet().to(device)
    # model3            = AttU_Net().to(device)
    # model4            = TransUNet(img_dim=256,in_channels=3,out_channels=128,head_num=4,mlp_dim=512,block_num=8,patch_dim=16,class_num=1)
    # model5            = SwinTransformerSys().to(device)
    # #model6           = Build_LeViT_UNet_192(num_classes=1, pretrained=True).to(device)
    
    # model  = BDFormer(img_size=256, in_channels=3, num_classes=1, window_size=8).to(device)
    # model = CFATransUnet().to(device)
    model = DTrAttUnet(in_channels=3,out_channels=1,img_size=224).to(device)
    # model = FCTNet().to(device)
    # model =  MedNeXtSegmentationModel().to(device)
    # model = ResponseFusionAttentionUConvNextBase().to(device)
    # model = TransAttUnet().to(device)

    checkpoint_path = ML_DATA_OUTPUT+str(model.__class__.__name__)+str(res)
    
    #data            = 'isic_2018_1' # to generalized the model giving different dataset. 
    trainable_params      = sum(p.numel() for p in model.parameters() if p.requires_grad)
    args.aug            = False
    args.shuffle        = True
    test_loader         = loader(args.mode, args.sslmode_modelname, args.train, args.bsize, args.workers, args.imsize, args.cutoutpr, args.cutoutbox, args.aug, args.shuffle, args.sratio, data)

    print(f"model path:",res)
    print('test_loader loader transform',test_loader.dataset.tr)
    print(f"Testing for Model  : {model.__class__.__name__}, model params: {trainable_params}")
    print(f"training with {len(test_loader)*args.bsize} images")

    try:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(checkpoint_path))
        else: 
            model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
            # model1.load_state_dict(torch.load(checkpoint_path1, map_location=torch.device('cpu')))
            # model2.load_state_dict(torch.load(checkpoint_path2, map_location=torch.device('cpu')))
            # model3.load_state_dict(torch.load(checkpoint_path3, map_location=torch.device('cpu')))
            # model4.load_state_dict(torch.load(checkpoint_path4, map_location=torch.device('cpu')))
            # model5.load_state_dict(torch.load(checkpoint_path5, map_location=torch.device('cpu')))
            # #model6.load_state_dict(torch.load(checkpoint_path6, map_location=torch.device('cpu')))

    except:
         raise Exception("******* No Checkpoint Path  *********")
    
    metrics_score = [ 0.0, 0.0, 0.0, 0.0, 0.0]
    model.eval()


    #TopoLoss      = Topological_Loss(lam=0.5).to(device)

    for batch in tqdm(test_loader, desc=f"testing ", leave=False):
        images,labels   = batch                
        # Desired output size, e.g., resizing to 128x128
        output_size = (224, 224)

        # Resize the entire batch
        resized_images = torch.stack([F.resize(image, output_size) for image in images])
       
        with torch.no_grad():

            model_output = process_model_output(model,images)
            # model_output1    = model1(images)
            # model_output2    = model2(images)
            # model_output3    = model3(images)
            # model_output4    = model4(images)
            # model_output5    = model5(resized_images)
            # #model_output6    = model6(resized_images)


            prediction          = torch.sigmoid(model_output)
            # prediction1       = torch.sigmoid(model_output1)
            # prediction2       = torch.sigmoid(model_output2)
            # prediction3       = torch.sigmoid(model_output3)
            # prediction4       = torch.sigmoid(model_output4)
            # prediction5       = torch.sigmoid(model_output5)
            #prediction6        = torch.sigmoid(model_output6)
            
            #topo_loss           = TopoLoss(model_output,labels)

            score = calculate_metrics(labels, prediction)
            metrics_score = list(map(add, metrics_score, score))

            acc = {    "jaccard"      : metrics_score[0]/len(test_loader),
                        "f1"          : metrics_score[1]/len(test_loader),
                        "recall"      : metrics_score[2]/len(test_loader),
                        "precision"   : metrics_score[3]/len(test_loader),
                        "acc"         : metrics_score[4]/len(test_loader)
                        }        

            print(f" Jaccard (IoU): {acc['jaccard']:1.4f} - F1(Dice): {acc['f1']:1.4f} - Recall: {acc['recall']:1.4f} - Precision: {acc['precision']:1.4f} - Acc: {acc['acc']:1.4f} ")

        #  PLOTTING  #

    args.shuffle = False

    test_loader    = loader(args.mode,args.sslmode_modelname,args.train, args.bsize,args.workers,args.imsize,args.cutoutpr,args.cutoutbox,args.aug,args.shuffle,args.sratio,data)
    
    columns = ["model_name","image", "pred", "target","Jaccard(IOU)"]
    image_table = wandb.Table(columns=columns) 

    for batch in test_loader:
        images,labels   = batch
        model_output    = model(images)
        
        print("plotting one set of figure")
        prediction      = torch.sigmoid(model_output)
        im,pred,lab     = log_image_table(images, prediction, labels, (len(test_loader.dataset)%args.bsize)-1)
        image_table.add_data(str(model.__class__.__name__)+'_'+str(res),wandb.Image(im),wandb.Image(pred),wandb.Image(lab),acc["jaccard"])
        break

    wandb.log({"predictions_table":image_table})
    wandb.log(acc)
    wandb.finish()    
