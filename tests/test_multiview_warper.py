import numpy as np
from dsynth.multiview_warper import MultiviewWarper
from dsynth.multiview_data_generators import tless

def test_multiview_warper(vis=False):

    mv_warpers = {}
    for obj_id in range(1,5):
        paths_to_masks = tless.generate_masks(obj_id=obj_id, num_processes=4, debug=True, overwrite=False)
        generator = tless.generate(obj_id=obj_id, paths_to_masks=paths_to_masks, debug=True)

        mv_warper = MultiviewWarper(generator)
        mv_warpers['obj_%02d' % obj_id] = mv_warper 

    for obj_id, warper in mv_warpers.items():
        path_to_rgb, path_to_mask = warper.paths_to_rgb_mask[0]
        R = warper.Rs[0]
        t = warper.ts[0]
        K = np.linalg.inv(warper.K_invs[0])

        W, view_id = warper.match_and_warp((R,t), K, (400,400), alpha=.8, light_setting=None)




