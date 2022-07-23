import torch
from torch import optim, nn, autograd
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import os
import argparse
import numpy as np
from model import Audio2FeatureModel
from pathlib import Path

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
                
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, default='/root/Datasets/Features/M003/audio/front/neutral/level_1/001.npy')
    parser.add_argument("--model_path", type=str, default='./saved_model/backups/audio2feature_790000.bak')
    parser.add_argument("--save_path", type=str, default='./result/lm_pred.npy')

    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument("--num_thread", type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    
    parser.add_argument('--input_ndim', type=int, default=13)
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
        
        audio_data = torch.tensor(np.load(opt.test_data_path).T).to(device).unsqueeze(0)
        print(audio_data.shape)
        
        landmark_preds = model(audio_data)      
        print(landmark_preds.shape)

        if not os.path.exists(opt.save_path):
            os.makedirs(os.path.dirname(opt.save_path), exist_ok=True)
        np.save(opt.save_path, landmark_preds.cpu().numpy())

