'''
This file implements MultiviewDataset interface for t-less dataset (http://cmp.felk.cvut.cz/t-less). 
  - See multiview_warper.py for ViewDataset interface.
  - See tests/test_tless.py for use case.

To use this class with unit_test=False mode, 
    1) One must download the dataset first, and specify the path to the dataset (tless_root_path)
    2) For 3D model, we use "reconst" models. User must convert the obj_xx.ply
    to (obj_xx.obj, obj_xx.obj.mtl, obj_xx_tex.png). Using meshlab is recommended.
'''

import os
import glob
import yaml
from dsynth import MultiviewDataset, ViewExample
from dsynth import pkg_root
from dsynth.util import grabcut_util
from dsynth.util import object_loader

_path_to_rgb      = 't-less_v2/train_{sensor}/{obj_id:02d}/rgb/{view_id}.png'
_path_to_gt_yml   = 't-less_v2/train_{sensor}/{obj_id:02d}/gt.yml'
_path_to_info_yml = 't-less_v2/train_{sensor}/{obj_id:02d}/info.yml'
_path_to_mask     = 't-less_v2/train_{sensor}/{obj_id:02d}/mask/{view_id}.png'
_path_to_obj      = 't-less_v2/models_reconst/obj_{obj_id:02d}.obj'


def path_to_test():
    return os.path.join(os.path.dirname(pkg_root), 'test_data')

default_generate_mask_settings = dict(
    num_processes = 4,
    overwrite = False,
    grabcut_with_pose_prior_kwargs = {
        'erosion_ratio': .6,
        'dilation_ratio': 1.4,
        'num_iter': 40,
    },

)

class TlessMultiviewDataset(MultiviewDataset):

    def __init__(self, 
        obj_id,
        tless_root_path='./', 
        sensor = 'primesense',
        # sensor = 'kinect',
        path_to_mask = None,
        generate_mask_settings = None,
        unit_test=False):

        '''
        tless_root_path:
            "t-lessv2/" must be under this path.
        path_to_mask:
            formatted string that has exactly one named placehoder, "{view_id}"
            e.g. "my_mask_folder/mask_{view_id}.png"
        '''

        if unit_test:
            assert obj_id == 2
            tless_root_path = path_to_test()

        if generate_mask_settings is None:
            generate_mask_settings = default_generate_mask_settings

        path_to_rgb = os.path.join(tless_root_path, _path_to_rgb)
        path_to_gt_yml = os.path.join(tless_root_path, _path_to_gt_yml).format(sensor=sensor, obj_id=obj_id)
        path_to_info_yml = os.path.join(tless_root_path, _path_to_info_yml).format(sensor=sensor, obj_id=obj_id)
        path_to_obj = os.path.join(tless_root_path, _path_to_obj).format(obj_id=obj_id)

        if path_to_mask is None:
            path_to_mask = os.path.join(tless_root_path, _path_to_mask)

        img_files = sorted( glob.glob(
                        os.path.join(os.path.dirname(path_to_rgb).format(sensor=sensor, obj_id=obj_id), '*.png')) )
        self.view_ids = view_ids = [os.path.basename(img_file).split('.')[0] for img_file in img_files]
        self.N = N = len(img_files)
        self.path_to_obj = path_to_obj


        def generate_view_examples():

            with open(path_to_gt_yml) as f:
                print('\nloading %s...' % path_to_gt_yml)
                gt = yaml.load(f) # contains groundtruth pose
                print('done.')
            with open(path_to_info_yml) as f:
                print('loading %s...' % path_to_info_yml)
                info = yaml.load(f) # contains intrinsics
                print('done.')

            for view_id in view_ids:
                k = int(view_id)
                _gt = gt[k]
                _info = info[k]

                R = _gt[0]['cam_R_m2c']
                t = _gt[0]['cam_t_m2c']
                K = _info['cam_K']

                rgb_file = path_to_rgb.format(sensor=sensor, obj_id=obj_id, view_id=view_id)
                try:
                    mask_file = path_to_mask.format(sensor=sensor, obj_id=obj_id, view_id=view_id)
                except:
                    mask_file = path_to_mask.format(view_id=view_id)

                yield ViewExample(view_id, (rgb_file, mask_file), R, t, K)
            
        self.view_examples = [view_info for view_info in generate_view_examples()]

        self.generate_mask(generate_mask_settings)


    def __getitem__(self, idx):
        '''
        Must return a ViewExample.
        '''
        return self.view_examples[idx]

    def __len__(self):
        return self.N


    def generate_mask(self, generate_mask_settings):
        mesh = object_loader.OBJFile(self.path_to_obj, None)

        num_processes = generate_mask_settings['num_processes']
        overwrite = generate_mask_settings['overwrite']
        grabcut_with_pose_prior_kwargs =  generate_mask_settings['grabcut_with_pose_prior_kwargs']

        grabcut_util.generate_masks_parallel(
            mesh.vertices, mesh.faces,
            self.view_examples,
            num_processes=num_processes, num_views = self.N, 
            overwrite=overwrite, grabcut_with_pose_prior_kwargs = grabcut_with_pose_prior_kwargs
            )

