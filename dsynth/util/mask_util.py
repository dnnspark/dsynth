import numpy as np
import cv2
from skimage.morphology import binary_erosion, binary_dilation
from dsynth.util.project_util import project_mesh_barebone

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


def grabcut_with_pose_prior(
                    I, vertices, faces, 
                    R, t, K,
                    erosion_ratio, dilation_ratio,
                    num_iter, vis=False):
    
    '''
    '''

    M0 = project_mesh_barebone(vertices, faces, R, t, K, I.shape)[:,:,0]

    erosion_radius = compute_radius(M0, erosion_ratio)
    dilation_radius = compute_radius(M0, dilation_ratio)

    surely_fg = binary_erosion(M0, np.ones((erosion_radius,erosion_radius)))
    surely_bg = np.logical_not( binary_dilation(M0, np.ones((dilation_radius,dilation_radius))) )

    mask = np.ones(I.shape[:2], np.uint8) * cv2.GC_PR_FGD
    mask[np.where(surely_fg)] = cv2.GC_FGD
    mask[np.where(surely_bg)] = cv2.GC_BGD

    cv2.grabCut(I, mask, None, np.zeros((1,65),np.float64), np.zeros((1,65), np.float64), num_iter, cv2.GC_INIT_WITH_MASK)
    M = np.logical_or(mask==cv2.GC_FGD, mask==cv2.GC_PR_FGD).astype(np.uint8) * 255

    if vis:
        plt.figure(1), plt.clf(), plt.imshow(I), plt.imshow(M, alpha=.5), 
        plt.title('projected')
        plt.figure(2), plt.clf(), plt.imshow(I), plt.imshow(surely_fg, alpha=.5), 
        plt.title('surely_fg')
        plt.figure(3), plt.clf(), plt.imshow(I), plt.imshow(surely_bg, alpha=.5), 
        plt.title('surely_bg')
        plt.show(block=False)
        plt.figure(4), plt.clf(), plt.imshow(I), plt.imshow(M, alpha=.5), plt.title('gc')
        plt.show(block=False)
        plt.pause(.03)

    return M

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
