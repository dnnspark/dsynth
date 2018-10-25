import numpy as np
import cv2
from skimage.morphology import binary_erosion, binary_dilation

import matplotlib.pyplot as plt

def compute_radius(M, alpha):
    ''' 
    eroding a box with area S by r results in a box of S*alpha area

    if alpha > 1, return dilation radius
    if 0< alpha < 1, return erosion radius
    '''
    assert M.ndim == 2
    if np.unique(M).sum() == 255:
        M = np.float32(M)/255.
    assert alpha > 0.
    S = M.sum()
    r = .5 * np.sqrt(S) * np.abs(1. - np.sqrt(alpha) )
    return int(np.round(r))

def fill_interior(M, radius=4):

    r = radius

    J = binary_dilation(M, np.ones((r,r))).astype(np.uint8)*255

    h,w = M.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    F = cv2.floodFill(J, seedPoint=(0,0), newVal=255, mask=mask)
    F = (1-F[2])*255
    F = F[1:-1,1:-1]
    F = binary_erosion(F, np.ones((r,r))).astype(np.uint8)*255

    return F
