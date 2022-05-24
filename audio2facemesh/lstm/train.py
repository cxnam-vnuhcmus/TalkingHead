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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/root/Data_Preprocessing')
    parser.add_argument('--mfcc_path', type=str, default='MFCC_MEAD')
    parser.add_argument('--mfcc_pickle', type=str, default='all_path_mfcc.pkl')
    parser.add_argument('--landmark_path', type=str, default='Mesh_light')
    parser.add_argument('--pretrain_path', type=str, default='./saved_model/audio2feature.pt')
    parser.add_argument("--save_path", type=str, default='./saved_model')
    parser.add_argument("--backup_path", type=str, default='./saved_model/backups')

    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument("--num_thread", type=int, default=10)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epoches', type=int, default=100)
    parser.add_argument("--learning_rate", type=int, default=1e-4)
    
    parser.add_argument('--use_pretrained', action="store_true")
    parser.add_argument("--save_every", type=int, default=20000)
    parser.add_argument("--backup_every", type=int, default=50000)

    parser.add_argument('--input_ndim', type=int, default=30*28)
    parser.add_argument('--output_ndim', type=int, default=478*3)
    return parser.parse_args()

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

if __name__ == '__main__': 
    opt = parse_args()
    device = torch.device(opt.device)
    if opt.device != 'cpu':
        torch.cuda.set_device(opt.device)
    
    training_set = MFCC_Dataset(opt)
    
    train_loader = DataLoader(training_set,
                            batch_size=opt.batch_size,
                            num_workers=opt.num_thread,
                            shuffle=True,
                            drop_last=True,
                            collate_fn=my_collate)
    
    model = Audio2FeatureModel(opt).to(device)
    
    optim = optim.Adam(model.parameters(), lr=opt.learning_rate*0.1)

    criteon = nn.MSELoss()
    
#     criteon = nn.KLDivLoss(reduction="batchmean", log_target= True)
    
    # Load any existing model
    if opt.use_pretrained:
        model.load(opt.pretrain_path, optim)
    else:
        print("Starting the training from scratch.")
    
    os.makedirs(opt.save_path, exist_ok = True)
    os.makedirs(opt.backup_path, exist_ok = True)
    steps_per_epoch = np.ceil(len(training_set) / opt.batch_size).astype(np.int32)
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    
    model.train()
    
    for epoch in range(opt.n_epoches):
        for i, (x,y) in enumerate(train_loader):
            start_time = time.time()            
            
            x, y = x.to(device), y.to(device)
            
            pred = model(x)
            
            pred = pred.reshape(x.shape[0], -1, y.shape[2])

#             pred = F.log_softmax(pred)
#             y = F.log_softmax(y)
            
            loss = criteon(pred, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            
            time_window.append(time.time() - start_time)
            loss_window.append(loss.item())
            step = model.get_step()
            
            msg = f"| Epoch: {epoch}/{opt.n_epoches} ({i}/{steps_per_epoch}) | Loss: {loss_window.average:#.4} | " \
                      f"{1./time_window.average:#.2} steps/s | Step: {step//1000}k | "
            stream(msg)
            
            # Overwrite the latest version of the model
            if opt.save_every != 0 and step % opt.save_every == 0:
                print("Saving the model (step %d)" % step)
                save_fpath = os.path.join(opt.save_path,'audio2feature.pt')
                model.save(save_fpath, optimizer=optim)

            # Make a backup
            if opt.backup_every != 0 and step % opt.backup_every == 0:
                print("Making a backup (step %d)" % step)
                backup_fpath = os.path.join(opt.backup_path, f'audio2feature_{step:06d}.bak')
                model.save(backup_fpath, optimizer=optim)
            