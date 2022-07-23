import os.path
from PIL import Image
import numpy as np
import json
import glob
from scipy.optimize import curve_fit
import warnings

def func(x, a, b, c):    
    return a * x**2 + b * x + c

def linear(x, a, b):
    return a * x + b

def setColor(im, yy, xx, color):
    if len(im.shape) == 3:
        if (im[yy, xx] == 0).all():            
            im[yy, xx, 0], im[yy, xx, 1], im[yy, xx, 2] = color[0], color[1], color[2]            
        else:            
            im[yy, xx, 0] = ((im[yy, xx, 0].astype(float) + color[0]) / 2).astype(np.uint8)
            im[yy, xx, 1] = ((im[yy, xx, 1].astype(float) + color[1]) / 2).astype(np.uint8)
            im[yy, xx, 2] = ((im[yy, xx, 2].astype(float) + color[2]) / 2).astype(np.uint8)
    else:
        im[yy, xx] = color[0]

def drawEdge(im, x, y, bw=1, color=(255,255,255), draw_end_points=False):
    if x is not None and x.size:
        h, w = im.shape[0], im.shape[1]
        # edge
        for i in range(-bw, bw):
            for j in range(-bw, bw):
                yy = np.maximum(0, np.minimum(h-1, y+i))
                xx = np.maximum(0, np.minimum(w-1, x+j))
                setColor(im, yy, xx, color)

        # edge endpoints
        if draw_end_points:
            for i in range(-bw*2, bw*2):
                for j in range(-bw*2, bw*2):
                    if (i**2) + (j**2) < (4 * bw**2):
                        yy = np.maximum(0, np.minimum(h-1, np.array([y[0], y[-1]])+i))
                        xx = np.maximum(0, np.minimum(w-1, np.array([x[0], x[-1]])+j))
                        setColor(im, yy, xx, color)

def interpPoints(x, y):    
    if abs(x[:-1] - x[1:]).max() < abs(y[:-1] - y[1:]).max():
        curve_y, curve_x = interpPoints(y, x)
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


def connect_keypoints(face_pts, face_list, size):
    w, h = size
    output_edges = np.zeros((h, w, 3), np.uint8)
    
    ### face
    edge_len = 2
    for edge_list in face_list:
        for edge in edge_list:
            for i in range(0, max(1, len(edge)-1), edge_len-1):             
                sub_edge = edge[i:i+edge_len]
                x, y = face_pts[sub_edge, 0], face_pts[sub_edge, 1]
                if 0 not in x:
                    curve_x, curve_y = interpPoints(x, y)
                    drawEdge(output_edges, curve_x, curve_y, draw_end_points=True)

    return output_edges

def define_edge_lists():
    ### pose        
    pose_edge_list = []
    pose_color_list = []

    pose_edge_list += [        
        [ 0,  1], [ 1,  8],                                         # body
        [ 1,  2], [ 2,  3], [ 3,  4],                               # right arm
        [ 1,  5], [ 5,  6], [ 6,  7],                               # left arm
        [ 8,  9], [ 9, 10], [10, 11], [11, 24], [11, 22], [22, 23], # right leg
        [ 8, 12], [12, 13], [13, 14], [14, 21], [14, 19], [19, 20]  # left leg
    ]
    pose_color_list += [
        [153,  0, 51], [153,  0,  0],
        [153, 51,  0], [153,102,  0], [153,153,  0],
        [102,153,  0], [ 51,153,  0], [  0,153,  0],
        [  0,153, 51], [  0,153,102], [  0,153,153], [  0,153,153], [  0,153,153], [  0,153,153],
        [  0,102,153], [  0, 51,153], [  0,  0,153], [  0,  0,153], [  0,  0,153], [  0,  0,153]
    ]

    ### hand
    hand_edge_list = [
        [0,  1,  2,  3,  4],
        [0,  5,  6,  7,  8],
        [0,  9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20]
    ]
    hand_color_list = [
        [204,0,0], [163,204,0], [0,204,82], [0,82,204], [163,0,204]
    ]

    ### face        
    face_list = [
                 [range(0, 17)], # face
                 [range(17, 22)], # left eyebrow
                 [range(22, 27)], # right eyebrow
                 [range(27, 31), range(31, 36)], # nose
                 [[36,37,38,39], [39,40,41,36]], # left eye
                 [[42,43,44,45], [45,46,47,42]], # right eye
                 [range(48, 55), [54,55,56,57,58,59,48]], # mouth
                 [range(60,65), [64,65,66,67,60]], #tongue
                ]
    return face_list,pose_edge_list, pose_color_list, hand_edge_list, hand_color_list