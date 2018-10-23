'''
MultiviewWarper implements an image-based rendering algorithm that uses multiview images of isolated object. 
Given a target camera matrix (i.e. intrinsics and extrnsics), the algorithm first find the 1-nearest neighbor view 
in the input view set. Then it computes 2D homography from the difference of target view and the matched input view,
and warp the input rgb and mask using the homography transformation. 

The distance metric used in nearest-neighbor is designed to measure the diffrence in
    1) normalized camera location, and 
    2) "look-at" vector
That is, if two cameras are radially projected on a same 3d location on a unit sphere, 
and if they are looking at a same location, they are considered equivalent 
in the nearest-neighbor match. This metric is roughly invariant to in-plane rotation and scaling. 

This works well for typical scenarios where the input views are from view-sphere image capture, 
and target view is looking at roughly the center of the scene.

As input, user must implement a python generator that yields a dict of:
    view_id: str
        unique id for the view
    paths_to_rgb_mask: (str, str) or {str: (str,str)}
        (path_to_rgb, path_to_mask) pair, or a dict of it.
        A single view may be associated with multiple images that are only different by, e.g., light setting.
        The dict is key'ed by the name of lighting setting.
    cam_R: 9-dim float
        rotation
    cam_t: 3-dim float
        translation
    cam_K: 9-dim float
        intrinsics
    (optional) path_to_obj: str
        path to .obj file, the 3d model of object.
        This must be same for all views, if any.

The camera coordinate is assumed to follow standard pinhole camera model:
    https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
'''

import numpy as np
import cv2
import abc
from imageio import imread
from dsynth.util.multiview_util import normalize, warp_from_camera_motion

class ViewDataset(abc.ABC):

    @abc.abstractmethod
    def __getitem(self, idx):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass



class MultiviewWarper():

    def __init__(self, view_generator, plane=None):

        self.init(view_generator)
        self.plane = plane

    def init(self, view_generator):
        '''
        Cache all necessary data of the src views in a numpy array,
        so that the match is performed by simple numpy operations.

        ids   : view_id's

        * Camera parameters
        Rs    : rotations
        ts    : translations
        K_invs: inverted intrinsics

    
        * Derived from (R,t). 
        L    : normalized camera locations in model's frame.
        Z    : normalized "look-at" vectors in model's frame.
            this is the last rows of rotation matrices.
        N    : 
        '''

        print('MultiviewWarper: Caching view data...')

        N = 5000

        ids = ''
        paths_to_rgb_mask = []
        Rs = np.zeros( (N, 3, 3), np.float32) 
        ts = np.zeros( (N,3), np.float32) 
        K_invs = np.zeros( (N,3,3), np.float32)
        L = np.zeros( (N,3), np.float32)
        Z = np.zeros( (N,3), np.float32)
        N = np.zeros( (N,3), np.float32)

        for k, view in enumerate(view_generator):
            ids += ' %s' % view['view_id']
            paths_to_rgb_mask.append( view['paths_to_rgb_mask'] )

            # camera parameters
            R = np.float32( view['cam_R'] ).reshape(3,3)
            t = np.float32( view['cam_t'] )
            K_inv = np.linalg.inv( np.float32(view['cam_K']).reshape(3,3) )
            Rs[k] = R
            ts[k] = t
            K_invs[k] = K_inv

            L[k] = normalize( -np.dot(R.T, t) )
            Z[k] = R[-1]

            normal = R[-1,:] - np.dot(R.T, t)
            N[k] = normalize( normal )

            # all views are defined wrt one 3d model.
            _path_to_obj = view.get('path_to_obj', None)
            if k == 0:
                path_to_obj = _path_to_obj
            assert path_to_obj == _path_to_obj

        k += 1

        ids = ids.split(' ')[1:]
        Rs = Rs[:k]
        ts = ts[:k]
        K_invs = K_invs[:k]
        L = L[:k]
        Z = Z[:k]
        N = N[:k]


        self.ids = ids
        self.paths_to_rgb_mask = paths_to_rgb_mask
        self.Rs = Rs
        self.ts = ts
        self.K_invs = K_invs
        self.L = L
        self.Z = Z
        self.N = N
        self.path_to_obj = path_to_obj

        print('done.')

    def match_and_warp(self, target_pose, target_K, img_shape, alpha=.8, light_setting=None):
        '''
        1. Match
        two match metrics:
            a) normalized camera locations
            b) camera z-axis
        final matching score = linear combination of a) and b).

        2. Warp
        '''

        # 1. normalized camera location.
        R1,t1 = target_pose
        location = normalize( -np.dot(R1.T, t1) )
        location_scores = np.dot(self.L, location)

        # 2. z_axis
        z_axis = R1[-1]
        # z_axis = target_pose[2,:3].reshape(3,1)
        z_axis_scores = np.dot(self.Z, z_axis)
        
        scores = alpha * location_scores + (1.-alpha) * z_axis_scores

        best = np.argmax(scores)

        _id = self.ids[best]
        paths_to_rgb_mask = self.paths_to_rgb_mask[best]
        I, M, id_suffix = self.retrieve_img(paths_to_rgb_mask, light_setting)
        R0, t0 = self.Rs[best], self.ts[best]
        K0_inv = self.K_invs[best]
        if self.plane is None:
            n = self.N[best]
            d = 0.
        else:
            n,d = plane

        H = warp_from_camera_motion(R0, t0, R1, t1, n, 0., target_K, K0_inv)

        wi = cv2.warpPerspective(I, H, (img_shape[1], img_shape[0]) )
        wm = cv2.warpPerspective(M, H, (img_shape[1], img_shape[0]), flags = cv2.INTER_LINEAR )

        W = np.concatenate([wi,np.expand_dims(wm, axis=-1)], axis=-1)

        view_id = _id + id_suffix
        return W, view_id

    def retrieve_img(self, paths_to_rgb_mask, light_setting):

        if light_setting is not None:
            path_to_rgb, path_to_mask = paths_to_rgb_mask[light_setting]
        else:
            path_to_rgb, path_to_mask = paths_to_rgb_mask

        I = imread(path_to_rgb)
        M = imread(path_to_mask)

        if light_setting is not None:
            suffix = '-' + light_setting
        else:
            suffix = ''

        return I, M, suffix

