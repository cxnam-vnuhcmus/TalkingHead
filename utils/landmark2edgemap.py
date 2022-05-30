# import cv2
import numpy as np
# import matplotlib.pyplot as plt
import warnings
from scipy.optimize import curve_fit
import os
from tqdm import tqdm
import numpy as np
from functools import partial
from multiprocessing.pool import Pool
import argparse
import pathlib
from pathlib import Path
from PIL import Image

part_list = [[range(0, 17)],                                   # face
             [range(17, 22)],                                  # left eyebrow
             [range(22, 27)],                                  # right eyebrow
             [range(27, 31), range(31, 36)],                   # nose
             [[36,37,38,39], [39,40,41,36]],                   # left eye
             [[42,43,44,45], [45,46,47,42]],                   # right eye
             [range(48, 55), [54,55,56,57,58,59,48], range(60, 65), [64,65,66,67,60]]] # mouth and tongue

def func(x, a, b, c):    
    return a * x**2 + b * x + c

def linear(x, a, b):
    return a * x + b

# Given the start and end points, interpolate to get a line.
def interp_points(x, y):    
    if abs(x[:-1] - x[1:]).max() < abs(y[:-1] - y[1:]).max():
        curve_y, curve_x = interp_points(y, x)
        if curve_y is None:
            return None, None
    else:        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")    
            if len(x) < 3:
                popt, _ = curve_fit(linear, x, y)
            else:
                popt, _ = curve_fit(func, x, y)                
                if abs(popt[0]) > 1:
                    return None, None
        if x[0] > x[-1]:
            x = list(reversed(x))
            y = list(reversed(y))
        curve_x = np.linspace(x[0], x[-1], (x[-1]-x[0]))
        if len(x) < 3:
            curve_y = linear(curve_x, *popt)
        else:
            curve_y = func(curve_x, *popt)
    return curve_x.astype(int), curve_y.astype(int)

# Set a pixel to the given color.
def set_color(im, yy, xx, color):
    if len(im.shape) == 3:
        if (im[yy, xx] == 0).all():            
            im[yy, xx, 0], im[yy, xx, 1], im[yy, xx, 2] = color[0], color[1], color[2]            
        else:            
            im[yy, xx, 0] = ((im[yy, xx, 0].astype(float) + color[0]) / 2).astype(np.uint8)
            im[yy, xx, 1] = ((im[yy, xx, 1].astype(float) + color[1]) / 2).astype(np.uint8)
            im[yy, xx, 2] = ((im[yy, xx, 2].astype(float) + color[2]) / 2).astype(np.uint8)
    else:
        im[yy, xx] = color[0]

# Set colors given a list of x and y coordinates for the edge.
def draw_edge(im, x, y, bw=1, color=(255,255,255), draw_end_points=False):
    if x is not None and x.size:
        h, w = im.shape[0], im.shape[1]
        # edge
        for i in range(-bw, bw):
            for j in range(-bw, bw):
                yy = np.maximum(0, np.minimum(h-1, y+i))
                xx = np.maximum(0, np.minimum(w-1, x+j))
                set_color(im, yy, xx, color)

        # edge endpoints
        if draw_end_points:
            for i in range(-bw*2, bw*2):
                for j in range(-bw*2, bw*2):
                    if (i**2) + (j**2) < (4 * bw**2):
                        yy = np.maximum(0, np.minimum(h-1, np.array([y[0], y[-1]])+i))
                        xx = np.maximum(0, np.minimum(w-1, np.array([x[0], x[-1]])+j))
                        set_color(im, yy, xx, color)
                        
def convert_landmark_to_edgemap(keypoints, size):   
    w, h = size
    edge_len = 3  # interpolate 3 keypoints to form a curve when drawing edges
    # edge map for face region from keypoints
    im_edges = np.zeros((h, w), np.uint8) # edge map for all edges
    im_edges_full = im_edges.copy()

    for edge_list in part_list:
        for edge in edge_list:
            for i in range(0, max(1, len(edge)-1), edge_len-1): # divide a long edge into multiple small edges when drawing
                sub_edge = edge[i:i+edge_len]
                x = keypoints[sub_edge, 0]
                y = keypoints[sub_edge, 1]

                curve_x, curve_y = interp_points(x, y) # interp keypoints to get the curve shape
                draw_edge(im_edges, curve_x, curve_y, bw=1)
                draw_edge(im_edges_full, curve_x, curve_y, bw=1)

#     fig = plt.figure(figsize=(15, 5))
#     ax = fig.add_subplot(1, 3, 1)
#     ax.imshow(image)
#     ax = fig.add_subplot(1, 3, 2)
#     ax.imshow(im_edges)
#     ax = fig.add_subplot(1, 3, 3)
#     ax.imshow(im_edges_full)
#     plt.show()
    
    return Image.fromarray(im_edges)



def process_landmark(path, save_path_edgemap=None):
    path_edgemap = pathlib.Path(save_path_edgemap).joinpath(*path.parts[5:-1])
    path_edgemap.mkdir(exist_ok=True, parents=True)
    
    file_name = path.parts[-1].replace('npy','jpg')

    lm_data = np.load(path, allow_pickle=True)
    
    if np.any(lm_data):
        lm_data = lm_data.astype(int)
    
        im_edge = convert_landmark_to_edgemap(lm_data, (540,540))

        save_edgemap = os.path.join(path_edgemap,file_name) 

        im_edge.save(save_edgemap)
    
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Landmark to Edgemap",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--landmark_path", type=str, default='/root/Data_Preprocessing/Landmark_MEAD')
    parser.add_argument("--edgemap_path", type=str, default='/root/Data_Preprocessing/Edgemap_MEAD')
    args = parser.parse_args() 
    
    names = [ name for name in os.listdir(args.landmark_path) if name.startswith('M') or name.startswith('W') ]
    names.sort()
    
    for name in names:
        path_lm = f'{args.landmark_path}/{name}/'
        save_path_edgemap = f'{args.edgemap_path}/{name}/'

        paths = []
        for path in tqdm(Path(path_lm).rglob('*.npy'), f"Load files {name}"):
            paths.append(path)
        
        for path in tqdm(paths, name, len(paths), unit="files"):
            path_edgemap = pathlib.Path(save_path_edgemap).joinpath(*path.parts[5:]).with_suffix('.jpg')
            if not os.path.exists(path_edgemap):
                process_landmark(path, save_path_edgemap)
        
#         func = partial(process_landmark, save_path_edgemap=save_path_edgemap)

#         job = Pool(20).imap(func, paths)
#         list(tqdm(job, name, len(paths), unit="files"))
    