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
from utils import *
from common_eval import evaluate_normalized_mean_error_torch
from tqdm import tqdm
from logger import Logger
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default='./datasets/training_data.txt')
    parser.add_argument('--test_data_path', type=str, default='./datasets/testing_data.txt')

    parser.add_argument("--save_path", type=str, default='./saved_model')
    parser.add_argument("--backup_path", type=str, default='./saved_model/backups')
    parser.add_argument("--log_path", type=str, default='./result')
    parser.add_argument("--save_every", type=int, default=50000)
    parser.add_argument("--backup_every", type=int, default=10000)
    parser.add_argument("--log_step", type=int, default=1000)

    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument("--num_thread", type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_epoches', type=int, default=500)
    parser.add_argument("--learning_rate", type=int, default=1e-4)
    
    parser.add_argument('--use_pretrained', type=bool, default=False)
    parser.add_argument('--pretrain_path', type=str, default='')

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
    
    train_dataset = A2L_Dataset(opt.train_data_path)
    
    train_dataloader = DataLoader(train_dataset,
                            batch_size=opt.batch_size,
                            num_workers=opt.num_thread,
                            shuffle=True,
                            drop_last=True,
                            collate_fn=my_collate)
    
    test_dataset = A2L_Dataset(opt.test_data_path)
    
    test_dataloader = DataLoader(test_dataset,
                            batch_size=opt.batch_size,
                            num_workers=opt.num_thread,
                            shuffle=True,
                            drop_last=True,
                            collate_fn=my_collate)
    
    model = Audio2FeatureModel(opt).to(device)
    
    optim = optim.Adam(model.parameters(), lr=opt.learning_rate)

    criteon = nn.MSELoss()
    
    # Load any existing model
    if opt.use_pretrained:
        model.load(opt.pretrain_path, optim)
    else:
        print("Starting the training from scratch.")
    
    os.makedirs(opt.save_path, exist_ok = True)
    os.makedirs(opt.backup_path, exist_ok = True)
    steps_per_epoch = np.ceil(len(train_dataset) / opt.batch_size).astype(np.int32)
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    
    model.train()
    
    logger = Logger(opt.log_path)
    
    for epoch in range(opt.n_epoches):
        for i, (x,y) in enumerate(train_dataloader):
            if x is not None:
                start_time = time.time()            

                x, y = x.to(device), y.to(device)

                pred = model(x)

#                 loss = evaluate_normalized_mean_error_torch(pred, y)
                loss = criteon(pred, y)

                optim.zero_grad()
                loss.backward()
                optim.step()

                # Logging.
                losses = {}
                losses['train_loss'] = loss.item()
                nrmse = evaluate_normalized_mean_error_torch(pred, y)
                losses['train_nrmse'] = nrmse.cpu()

                time_window.append(time.time() - start_time)
                loss_window.append(loss)
                step = model.get_step()

                msg = f"| Epoch: {epoch}/{opt.n_epoches} ({i}/{steps_per_epoch}) | Loss: {loss_window.average:#.4} " \
                      f"| NRME: {losses['train_nrmse']:#.4} | " \
                      f"{1./time_window.average:#.2} steps/s | Step: {step//1000}k | "
                logger.log(msg,stdout=True)

                # Print out training information.
                if step % opt.log_step == 0:
                    for tag, value in losses.items():
                        logger.scalar_summary(tag, value, step+1)

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
