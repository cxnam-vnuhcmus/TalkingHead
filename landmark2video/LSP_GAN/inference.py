import os
import subprocess
from os.path import join
from tqdm import tqdm
import numpy as np
import torch
from collections import OrderedDict
import librosa
from skimage.io import imread
import cv2
import scipy.io as sio
import argparse
import yaml
import albumentations as A
import albumentations.pytorch
from pathlib import Path

from options.test_feature2face_options import TestOptions as RenderOptions

from datasets import create_dataset
from models import create_model
from models.networks import APC_encoder
import util.util as util
from util.visualizer import Visualizer
from funcs import utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edgemap_path', type=str, default='/root/Data_Preprocessing/Edgemap_MEAD/M003/front/angry/level_1/001/000.jpg')
    parser.add_argument('--landmark_path', type=str, default='/root/Data_Preprocessing/Landmark_MEAD/M003/front/angry/level_1/001/010.npy')
    parser.add_argument('--sample_landmark_path', type=str, default='./sample_pred_landmark.npy')

    parser.add_argument('--id', default='Obama1', help="person name, e.g. Obama1, Obama2, May, Nadella, McStay")
    parser.add_argument('--driving_audio', default='./data/Input/00083.wav', help="path to driving audio")
    parser.add_argument('--save_intermediates', default=0, help="whether to save intermediate results")
    parser.add_argument('--device', type=str, default='cpu', help='use cuda for GPU or use cpu for CPU')
    
    return parser.parse_args()

if __name__ == '__main__': 
    opt = parse_args()
    device = torch.device(opt.device)
    if opt.device != 'cpu':
        torch.cuda.set_device(opt.device)
        
    with open(join('./config/', opt.id + '.yaml')) as f:
        config = yaml.safe_load(f)
    data_root = join('./data/', opt.id)
    # create the results folder
    audio_name = os.path.split(opt.driving_audio)[1][:-4]
    save_root = join('./results/', opt.id, audio_name)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
        
    # candidates images    
    img_candidates = []
    for j in range(4):
        output = imread(join(data_root, 'candidates', f'normalized_full_{j}.jpg'))
        output = A.pytorch.transforms.ToTensor(normalize={'mean':(0.5,0.5,0.5), 
                                                          'std':(0.5,0.5,0.5)})(image=output)['image']
        img_candidates.append(output)
    img_candidates = torch.cat(img_candidates).unsqueeze(0).to(device) 
    
    Renderopt = RenderOptions().parse()
    Renderopt.dataroot = config['dataset_params']['root']
    Renderopt.load_epoch = config['model_params']['Image2Image']['ckp_path']
    Renderopt.size = config['model_params']['Image2Image']['size']
    ## GPU or CPU
    if opt.device == 'cpu':
        Renderopt.gpu_ids = []
    else:
        Renderopt.gpu_ids = [torch.cuda.current_device()]

    print('---------- Loading Model: {} -------------'.format(Renderopt.task))
    facedataset = create_dataset(Renderopt) 
    Feature2Face = create_model(Renderopt)
    Feature2Face.setup(Renderopt)   
    Feature2Face.eval()
    visualizer = Visualizer(Renderopt)


    fit_data = np.load(config['dataset_params']['fit_data_path'])
    ref_trans = fit_data['trans'][:,:,0].astype(np.float32)[1]
    shoulder3D = np.load(join(data_root, 'shoulder_points3D.npy'))[1]
    shoulder_AMP = config['model_params']['Headpose']['shoulder_AMP']
    std_mean_pts3d = np.load(config['dataset_params']['pts3d_path']).mean(axis=0)
    camera = utils.camera()
    camera_intrinsic = np.load(join(data_root, 'camera_intrinsic.npy')).astype(np.float32)
    mean_pts3d = np.load(join(data_root, 'mean_pts3d.npy'))
    pts3d = np.load(config['dataset_params']['pts3d_path']) + mean_pts3d

    # project
    project = camera_intrinsic.dot(shoulder3D.T)
    project[:2, :] /= project[2, :]  # divide z
    pred_shoulders = project[:2, :].T
    
    
    print(camera_intrinsic)

    lm_data = np.load(opt.landmark_path, allow_pickle=True) 
    lm_data[:, 0] -= 20
    sample_lm = np.load(opt.sample_landmark_path, allow_pickle=True)
    sample_lm[46, :] = lm_data[60]
    sample_lm[47:52, :] = lm_data[49:54]
    sample_lm[52, :] = lm_data[64]
    sample_lm[53:58, :] = lm_data[55:60]
    sample_lm[58:61, :] = lm_data[61:64]
    sample_lm[61:64, :] = lm_data[65:68]
    
    input_feature_maps = facedataset.dataset.draw_face_feature_maps(sample_lm) 
    
#     input_feature_maps = cv2.imread(opt.edgemap_path, cv2.IMREAD_GRAYSCALE)
#     input_feature_maps = cv2.resize(input_feature_maps, (512, 512), interpolation = cv2.INTER_AREA)    

    input_feature_maps = facedataset.dataset.draw_shoulder_points(input_feature_maps, pred_shoulders)    
    
    
    
    (thres, input_feature_maps) = cv2.threshold(input_feature_maps, 128, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    input_feature_maps = input_feature_maps.astype(np.float32)
    
    input_feature_maps = torch.tensor(input_feature_maps).unsqueeze(0).unsqueeze(0).to(device)
    
    #### 6. Image2Image translation & Save resuls
#     input_feature_maps = np.load('/root/TalkingHead/landmark2video/LSP_GAN/sample_map.npy')        
#     input_feature_maps = torch.tensor(input_feature_maps).unsqueeze(0).to(device)

    pred_fake = Feature2Face.inference(input_feature_maps, img_candidates) 
    # save results
    visual_list = [('pred', util.tensor2im(pred_fake[0]))]
    visual_list += [('input', np.uint8(input_feature_maps[0][0].cpu().numpy() * 255))]
    visuals = OrderedDict(visual_list)
    visualizer.save_images(save_root, visuals, str(0))


    print('Finish!')
