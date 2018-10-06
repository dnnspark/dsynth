import numpy as np
import cv2

def draw_mask(verts_2D, faces, im_shape, morph=None, kernel=np.ones((3,3))):
    """Returns a mask image of the shape im_shape.

    points are defined in verts_2D and faces defines how the points are connected.

    Parameters
    ----------
    verts_2D: np.array (N, 2)
      all the vertices
    faces: [object_loader.Face]
      define the faces of the mesh
    im_shape: tuple
      (h, w, 1) or (h, w, 3)
    morph: str
      Morphological transformation to apply to the segmentation mask
      "erode" - Erodes an image by convolving with kernel and only keeping pixels where all are 1s
      "dilate" - Dilates img by convolving with kernel keeping pixels where any are 1s
      "open" - Applies an erosion followed by a dilation. Useful for removing noise
      "close" - Applies a dilation then an erosion. Useful for removing holes
    kernel: np.array
      Kernel to be used for the morphological transformation.

    Returns
    -------
    mask: np.array
      (h, w, 1) or (h, w, 3) according to the input shape
      mask area has value 255; otherwise 0

    """
    SUPPORTED_MORPHS = {'erode': cv2.erode,
                        'dilate': cv2.dilate,
                        'open': lambda i, k: cv2.morphologyEx(i, cv2.MORPH_OPEN, k,),
                        'close': lambda i, k: cv2.morphologyEx(i, cv2.MORPH_CLOSE, k),
                        None: lambda i, k: i
                       }
    morph_func = SUPPORTED_MORPHS.get(morph, False)
    if not morph_func:
        print("Morph %s is not implemented" % morph)
        raise NotImplementedError

    # Create mask with same dimensions as img
    mask = np.zeros(im_shape, dtype=np.uint8)
    verts = np.rint(verts_2D).astype(np.int32)

    # Use the lines connecting projected vertices to construct a mask
    for face in faces:
        # Draw triangle
        pts = np.array([verts[face.i1], verts[face.i2], verts[face.i3]])
        cv2.fillConvexPoly(mask, pts, (255, 255, 255))

    return morph_func(mask, kernel)

def project_points(points_3D, R, t, K):
    """
    p = K[R|t]P
    points_3D: (N,3)
    R: (3,3)
    t: (3,)
    K: (3,3)
    """

    C = np.dot( K, np.concatenate([R, np.expand_dims(t,-1)], axis=1))
    P = np.concatenate( [points_3D.T, np.ones((1,points_3D.shape[0])) ], axis=0)
    p_h = np.dot(C, P)
    p = np.stack( [p_h[0]/p_h[-1], p_h[1]/p_h[-1]], axis=0).T

    return p

def project_mesh(mesh, R, t, K, img_shape):

    verts_3D = np.array(mesh.vertices)
    p = project_points(verts_3D, R, t, K)
    mask = draw_mask(p, mesh.faces, img_shape)

    return mask

def project_mesh_barebone(vertices, faces, R, t, K, img_shape):
    '''
    mesh object cannot be pickled, so cannot be used with multiprocessing.pool
    '''

    verts_3D = np.array(vertices)
    p = project_points(verts_3D, R, t, K)
    mask = draw_mask(p, faces, img_shape)

    return mask
