'''
This file implements view_generator interface for t-less dataset.
  - See multiview_ibr.py for view_generator interface.
  - See tests/test_multiview_ibr.py for use case.

Masks are generated from image in on-demand basis, using grabcut algorithm with ground truth pose as prior.
Using for-loop, it takes about 1.7s x 1300 = ~40 mins. To speed up this part, 
it runs on parallel using multiprocessing library of python.

'''
import os
import sys
import numpy as np
from imageio import imread, imwrite
import yaml
from dsynth.util import object_loader
from dsynth.util import mask_util
import multiprocessing as mp
import functools
import collections
import time
import tempfile


ViewInfo = collections.namedtuple('ViewInfo', 'path_to_rgb R t K path_to_mask')

# User must set this part.
PATH_TO_TLESS_ROOT = '/cluster/storage/dpark/t-less/t-less_v2'
SENSOR = 'primesense'
# SENSOR = 'kinect'

_gt_file = os.path.join(PATH_TO_TLESS_ROOT, 'train_{}/%02d/gt.yml'.format(SENSOR))
_info_file = os.path.join(PATH_TO_TLESS_ROOT, 'train_{}/%02d/info.yml'.format(SENSOR))
_obj_file = os.path.join(PATH_TO_TLESS_ROOT, 'models_reconst/obj_%02d.obj')

_view_id = '%04d'
_path_to_rgb = os.path.join(PATH_TO_TLESS_ROOT, 'train_{}/%02d/rgb/%04d.png'.format(SENSOR))
_path_to_mask = os.path.join(PATH_TO_TLESS_ROOT, 'train_{}/%02d/mask/%04d.png'.format(SENSOR))

N = 1296 

grabcut_with_pose_prior_kwargs = {
    'erosion_ratio': .6,
    'dilation_ratio': 1.4,
    'num_iter': 40,
}

py_version = sys.version_info[0]
if py_version < 3:
    file_not_found_error = IOError
else:
    file_not_found_error = FileNotFoundError



def load_yaml_files(obj_id):
    '''
    T-less dataset comes with two yaml files for each object.
    gt.yml contains ground-truth camera pose (model-to-camera)
    info.yml contains ground-truth camera intrinsics.
    '''

    with open(_gt_file % obj_id) as f:
        gt = yaml.load(f) # contains groundtruth pose
    with open(_info_file % obj_id) as f:
        info = yaml.load(f) # contains intrinsics

    return gt, info

def load_3d_model(obj_id):
    '''
    Load .obj file (mesh), and return only "vertices" and "faces" fields, 
    the minimum info for projecting the 3d model to mask.
    Mesh object itself cannot be pickled, which is required by use of multiprocessing.Pool
    '''

    obj_file = _obj_file % obj_id
    mesh = object_loader.OBJFile(obj_file, None)

    return mesh.vertices, mesh.faces, obj_file

def generate_view_info(obj_id, debug):

    gt, info = load_yaml_files(obj_id)

    num_views = 5 if debug else N

    for k in range(num_views):
        _gt = gt[k]
        _info = info[k]

        R = _gt[0]['cam_R_m2c']
        t = _gt[0]['cam_t_m2c']
        K = _info['cam_K']

        path_to_rgb = _path_to_rgb % (obj_id, k)
        path_to_mask = _path_to_mask % (obj_id, k)
        if debug:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as f:
                path_to_mask = f.name

        yield ViewInfo(path_to_rgb, R, t, K, path_to_mask)

def generate_and_write_mask(path_to_rgb, R, t, K, path_to_mask, vertices, faces, overwrite):
    '''
    Shallow wrapper for mask_util.brabcut_with_pose_prior. 
        - create parent directory, if not exists
        - create mask and save it on thie disk if
            1) it does not already exist, or 
            2) overwrite is set to True
    '''

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

        # 1. grabcut with pose as prior
        M = mask_util.grabcut_with_pose_prior(
                        I, vertices, faces, R, t, K, 
                        **grabcut_with_pose_prior_kwargs)

        imwrite(path_to_mask, M)

    return path_to_mask



def generate_masks(obj_id, num_processes=None, debug=False, overwrite=False):

    vertices, faces, _ = load_3d_model(obj_id)
    _generate_and_write_mask = functools.partial(generate_and_write_mask, vertices=vertices, faces=faces, overwrite=overwrite)
    view_info_generator = generate_view_info(obj_id, debug=debug)

    if num_processes is None:
        paths_to_masks = []
        for view_info in view_info_generator:
            paths_to_masks.append( _generate_and_write_mask(*view_info) )
    else:
        mp.set_start_method('spawn', force=True)
        chunksize = N // num_processes
        with mp.Pool(num_processes) as pool:
            paths_to_masks = pool.starmap(_generate_and_write_mask, view_info_generator, chunksize=chunksize)

    return paths_to_masks

def generate(obj_id, paths_to_masks=None, debug=False):

    _, __, obj_file = load_3d_model(obj_id)

    view_info_generator = generate_view_info(obj_id, debug)

    # path_to_obj
    _, __, path_to_obj = load_3d_model(obj_id)

    for k, view_info in enumerate(view_info_generator):

        path_to_rgb, R, t, K, path_to_mask = view_info

        # view_id
        view_id = '%04d' % k

        # paths_to_rgb_mask
        if paths_to_masks is not None:
            path_to_mask = paths_to_masks[k]
        paths_to_rgb_mask = (path_to_rgb, path_to_mask)

        out = {
            'view_id': view_id,
            'paths_to_rgb_mask': paths_to_rgb_mask,
            'cam_R': R,
            'cam_t': t,
            'cam_K': K,
            'path_to_obj': path_to_obj,
        }

        yield out


