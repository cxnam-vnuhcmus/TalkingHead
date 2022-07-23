import torch
from torch.utils.data import Dataset
import pickle
import os
import glob
import numpy as np
from pathlib import Path
import json

class A2L_Dataset(Dataset):
    def __init__(self, path):
        self.data_path = []
        with open(path, 'r') as f:
            for line in f:
                self.data_path.append(line.strip())

    def __getitem__(self, index):
        parts = self.data_path[index].split('|')
        audio_path, video_path = parts[0], parts[1]
        
        audio_data = np.load(audio_path).T # (Frame, 13)
        video_data = []
        del_rows = []
        for i in range(len(os.listdir(video_path))):
            path = os.path.join(video_path, f'{i}.json')
            with open(path) as json_file:
                data = json.load(json_file)
            if len(data['landmark']) == 68:
                video_data.append(np.asarray(data['landmark']).reshape(-1))
            else:
                del_rows.append(i)
        audio_data = np.delete(audio_data, del_rows, 0)
        video_data = np.asarray(video_data,dtype=np.float32) # (Frame, 68*2)
                    
        audio_data = torch.tensor(audio_data, dtype=torch.float)
        video_data = torch.tensor(video_data, dtype=torch.float)
        
        return (audio_data, video_data)
        
#         mfcc_fpath = os.path.join(self.data_root, self.mfcc_fpaths[index].strip())   
#         landmark_parts = self.mfcc_fpaths[index].strip().split('/')[1:]
#         landmark_parts.insert(1, 'front')
#         landmark_fpath = os.path.join(self.landmark_folder, *landmark_parts)
        
#         try:
#             if os.path.exists(landmark_fpath):
#                 mfcc_datas = [];
#                 for path in sorted(Path(mfcc_fpath).rglob('*.npy')):
#                     mfcc_data_np = np.load(os.path.join(mfcc_fpath, path))
#                     if mfcc_data_np.size == 1:
#                         return None     
#                     mfcc_data = torch.tensor(mfcc_data_np, dtype=torch.float)
#                     mfcc_data = mfcc_data.reshape(-1)
#                     mfcc_datas.append(mfcc_data)

#                 landmark_datas = [];
#                 for path in sorted(Path(landmark_fpath).rglob('*.npy')):
#                     landmark_data_np = np.load(path, allow_pickle=True)
#                     if landmark_data_np.size == 1:
#                         return None
#                     landmark_data = torch.tensor(landmark_data_np, dtype=torch.float)
#                     landmark_data = landmark_data.reshape(-1)
#                     landmark_datas.append(landmark_data)

#                 mfcc_datas = np.stack(mfcc_datas)
#                 landmark_datas = np.stack(landmark_datas)

#                 return (mfcc_datas, landmark_datas) # (BS, Frame, 30*28) - (BS,Frame, 68*2)

#             else:
#                 return None
#         except:
#             print(landmark_fpath)
#             return None
    
    def __len__(self):
        return len(self.data_path)
        
