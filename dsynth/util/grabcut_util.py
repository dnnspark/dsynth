import os
import numpy as np
import functools
import multiprocessing as mp
from imageio import imread, imwrite
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


# generate_and_write_mask() and generate_masks_parallel() are helper functions of grabcut_with_pose_prior()


file_not_found_error = FileNotFoundError # IOError for py2.
default_grabcut_with_pose_prior_kwargs = {
    'erosion_ratio': .6,
    'dilation_ratio': 1.4,
    'num_iter': 40,
    }
def generate_and_write_mask(
        path_to_rgb, path_to_mask,
        R, t, K, 
        vertices, faces, 
        overwrite, 
        grabcut_with_pose_prior_kwargs=None
    ):
    '''
    Shallow wrapper for grabcut_with_pose_prior()
        - create parent directory, if not exists
        - create mask and save it on thie disk if
            1) it does not already exist, or 
            2) overwrite is set to True
    '''

    if grabcut_with_pose_prior_kwargs is None:
        grabcut_with_pose_prior_kwargs = default_grabcut_with_pose_prior_kwargs

    R = np.float32(R).reshape(3,3)
    t = np.float32(t)
    K = np.float32(K).reshape(3,3)

    dirname = os.path.dirname(path_to_mask)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    try:
        imread(path_to_mask)
        if overwrite:
            raise file_not_found_error
    except file_not_found_error:
        I = imread(path_to_rgb)

        M = grabcut_with_pose_prior(
                        I, vertices, faces, R, t, K, 
                        **grabcut_with_pose_prior_kwargs)

        imwrite(path_to_mask, M)

    return path_to_mask
    
def generate_masks_parallel(
        vertices, faces,
        view_examples,
        num_processes=None, num_views = None, 
        overwrite=False,
        light_setting=None,
        grabcut_with_pose_prior_kwargs=None):
    '''
    Entry point for (parallel) mask generation.

    view_examples: [dsynth.ViewExample]
    num_processes: int or None
        if None, no multiprocessing
    light_setting: str or None
        if None, ViewExample.paths_to_rgb is (path_to_rgb, path_to_mask)
        else, {light_setting: (path_to_rgb, path_to_mask)}
    '''

    if grabcut_with_pose_prior_kwargs is None:
        grabcut_with_pose_prior_kwargs = default_grabcut_with_pose_prior_kwargs

    _generate_and_write_mask = functools.partial(generate_and_write_mask, 
                                    vertices=vertices, faces=faces, overwrite=overwrite, 
                                    grabcut_with_pose_prior_kwargs=grabcut_with_pose_prior_kwargs)

    def example_to_arg(view_example):
        if light_setting is not None:
            path_to_rgb, path_to_mask = view_example.paths_to_rgb_mask[light_setting]
        else:
            path_to_rgb, path_to_mask = view_example.paths_to_rgb_mask

        R = view_example.cam_R
        t = view_example.cam_t
        K = view_example.cam_K

        return [path_to_rgb, path_to_mask, R, t, K]


    if num_processes is None:
        paths_to_masks = []
        for view_example in view_examples:
            one_arg = example_to_arg(view_example)
            paths_to_masks.append( _generate_and_write_mask(*one_arg) )
    else:
        assert num_views is not None
        all_args = [example_to_arg(ex) for ex in view_examples]
        chunksize = max(1, num_views // num_processes )
        mp.set_start_method('spawn', force=True)
        with mp.Pool(num_processes) as pool:
            paths_to_masks = pool.starmap(_generate_and_write_mask, all_args, chunksize=chunksize)

    return paths_to_masks
