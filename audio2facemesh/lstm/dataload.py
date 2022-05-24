import torch
from torch.utils.data import Dataset
import pickle
import os
import glob
import numpy as np

class MFCC_Dataset(Dataset):
    def __init__(self, opt):
        mfcc_pickle_path = os.path.join(opt.data_root, opt.mfcc_path, opt.mfcc_pickle)
        with open(mfcc_pickle_path, 'rb') as f:
            self.mfcc_fpaths = pickle.load(f)
        self.landmark_folder = os.path.join(opt.data_root, opt.landmark_path)
        self.data_root = opt.data_root
        
    def __getitem__(self, index):
        mfcc_fpath = os.path.join(self.data_root, self.mfcc_fpaths[index].strip())        
        landmark_parts = self.mfcc_fpaths[index].strip().split('/')[1:]
        landmark_parts.insert(1, 'front')
        landmark_fpath = os.path.join(self.landmark_folder, *landmark_parts)
        
        try:
            if os.path.exists(landmark_fpath):
                mfcc_data = torch.tensor(np.load(mfcc_fpath), dtype=torch.float)
                landmark_npdata = np.load(landmark_fpath, allow_pickle=True)
                if landmark_npdata.size > 1:
                    landmark_data = torch.tensor(landmark_npdata, dtype=torch.float)
                    return (mfcc_data, landmark_data)
                else:
                    return None
            else:
                return None
        except:
            print(landmark_fpath)
    
    def __len__(self):
        return len(self.mfcc_fpaths)
        
