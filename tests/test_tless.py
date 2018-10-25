from dsynth.view_datasets.tless import TlessMultiviewDataset
from dsynth import MultiviewWarper
import numpy as np

def test_tless_dataset():
    dataset = TlessMultiviewDataset(obj_id=2, unit_test=True)
    ibr = MultiviewWarper(dataset)

    R = np.reshape(dataset[1].cam_R, (3,3)).astype(np.float32) 
    t = np.float32(dataset[1].cam_t)
    K = np.reshape(dataset[1].cam_K, (3,3)).astype(np.float32)
    W, view_id = ibr.match_and_warp( (R,t), K, (400,400))
    assert view_id == '0400'

    