import torch
from torch import optim, nn, autograd
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import os
import argparse
import numpy as np
from dataload import A2L_Dataset
from model import Audio2FeatureModel
from common_eval import evaluate_normalized_mean_error_torch
import sys
import statistics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, default='./datasets/testing_data.txt')
    parser.add_argument("--model_path", type=str, default='./saved_model/backups/audio2feature_790000.bak')

    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument("--num_thread", type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    
    parser.add_argument('--input_ndim', type=int, default=13)
    parser.add_argument('--output_ndim', type=int, default=68*2)
    return parser.parse_args()

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def pad1d(x, max_len, pad_value=0):
    return np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=pad_value)

def pad2d(x, max_len, pad_value=0):
    return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), mode="constant", constant_values=pad_value)

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) 
    
#     if len(batch) > 0:
#         # Text
#         x_lens = [len(x[0]) for x in batch]
#         max_x_len = max(x_lens)

#         chars = [pad2d(x[0].T, max_x_len).T for x in batch]
#         chars = np.stack(chars)


#         # Mel spectrogram
#         spec_lens = [x[1].shape[0] for x in batch]
#         max_spec_len = max(spec_lens)
#         mel_pad_value = -1 * 4.

#         mel = [pad2d(x[1].T, max_spec_len, pad_value=mel_pad_value).T for x in batch]
#         mel = np.stack(mel)

#         # Convert all to tensor
#         chars = torch.tensor(chars)
#         mel = torch.tensor(mel)

#         return chars, mel
#     else:
#         return None, None

if __name__ == '__main__': 
    opt = parse_args()
    device = torch.device(opt.device)
    if opt.device != 'cpu':
        torch.cuda.set_device(opt.device)
    
    test_dataset = A2L_Dataset(opt.test_data_path)
    
    test_dataloader = DataLoader(test_dataset,
                            batch_size=opt.batch_size,
                            num_workers=opt.num_thread,
                            shuffle=True,
                            drop_last=True,
                            collate_fn=my_collate)
    
    with torch.no_grad():
        model = Audio2FeatureModel(opt).to(device)
        model.load(opt.model_path)
        model.eval()

        total_loss = []
        for i, (x,y) in enumerate(test_dataloader):
            if x is not None:
                x, y = x.to(device), y.to(device)

                pred = model(x)

                nrmse = evaluate_normalized_mean_error_torch(pred, y)
                
                total_loss.append(nrmse.cpu().item())

                msg = f"| Step: ({i}/{len(test_dataset)}) " \
                      f"| NRME: {total_loss[-1]:#.4} | "
                      
                sys.stdout.write(f"\r{msg}")
        
        print(f"\nAverage loss: {statistics.mean(total_loss)}")
