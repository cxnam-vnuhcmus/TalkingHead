import torch
from torch import optim, nn, autograd
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import os
import argparse
import numpy as np
from display import *
from dataload import MFCC_Dataset
from model import Audio2FeatureModel
from utils import ValueWindow

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
                
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/root/Data_Preprocessing')
    parser.add_argument('--mfcc_path', type=str, default='MFCC_MEAD/M003/angry/level_1/001/000.npy')
    parser.add_argument('--face_path', type=str, default='Crop_MEAD/M003/front/angry/level_1/001/000.jpg')
    parser.add_argument('--landmark_path', type=str, default='Landmark_MEAD/M003/front/angry/level_1/001/000.npy')
    parser.add_argument('--model_path', type=str, default='./saved_model/audio2feature.pt')
#     parser.add_argument('--model_path', type=str, default='./saved_model/backups/audio2feature_1000000.bak')
    parser.add_argument("--save_path", type=str, default='./result/lm_pred.npy')

    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument("--num_thread", type=int, default=10)

    parser.add_argument('--input_ndim', type=int, default=30*28)
    parser.add_argument('--output_ndim', type=int, default=68*2)
    return parser.parse_args()

if __name__ == '__main__': 
    opt = parse_args()
    device = torch.device(opt.device)
    if opt.device != 'cpu':
        torch.cuda.set_device(opt.device)
    
    with torch.no_grad():
        model = Audio2FeatureModel(opt).to(device)
        model.load(opt.model_path)
        model.eval()

        mfcc_data = np.load(os.path.join(opt.data_root, opt.mfcc_path))
        mfcc_data = torch.tensor(np.expand_dims(mfcc_data, 0), dtype=torch.float).to(device)

        landmark_pred = model(mfcc_data)      
        landmark_pred = landmark_pred.squeeze(0)

        if not os.path.exists(opt.save_path):
            os.makedirs(os.path.dirname(opt.save_path), exist_ok=True)
        np.save(opt.save_path, landmark_pred.cpu().numpy())

#         landmark_real = np.load(os.path.join(opt.data_root, opt.landmark_path))
#         landmark_real = torch.tensor(landmark_real, dtype=torch.float).to(device)
# 
#         landmark_pred = F.log_softmax(landmark_pred)
#         landmark_real = F.softmax(landmark_real)
#         criteon = nn.KLDivLoss(reduction="batchmean")
#         loss = criteon(landmark_pred,landmark_real)

#         print(f"Loss: {loss}")

            