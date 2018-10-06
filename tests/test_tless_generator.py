import os
from dsynth.multiview_data_generators.tless import generate_masks, generate

def cleanup(paths_to_masks):
    for path_to_mask in paths_to_masks:
        os.remove(path_to_mask)

def test_generate_masks():
    paths_to_masks = generate_masks(obj_id=1, num_processes=None, debug=True, overwrite=False)

def test_generate_masks_parallel():
    paths_to_masks = generate_masks(obj_id=1, num_processes=4, debug=True, overwrite=False)
    cleanup(paths_to_masks)
    paths_to_masks = generate_masks(obj_id=2, num_processes=4, debug=True, overwrite=False)
    cleanup(paths_to_masks)

def test_generate_view_info_dict_as_input_of_multiview_ibr():
    paths_to_masks = generate_masks(obj_id=1, num_processes=4, debug=True, overwrite=False)
    generator = generate(obj_id=1, paths_to_masks=paths_to_masks, debug=True)

    fields = [
        'view_id',
        'paths_to_rgb_mask',
        'cam_R',
        'cam_t',
        'cam_K',
        'path_to_obj',
    ]

    for view_info in generator:
        for field in fields:
            assert field in view_info

    cleanup(paths_to_masks)
