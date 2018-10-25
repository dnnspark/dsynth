import numpy as np

def _col(vec):
    '''
    Convert to column vector, i.e. (N,1)
    '''

    if vec.ndim == 1:
        return np.expand_dims(vec, -1)
    else:
        assert vec.ndim == 2
        assert vec.shape[1] == 1
        return vec

def _row(vec):
    return _col(vec).T

def _vec(vec):
    assert vec.ndim == 2
    assert vec.shape[0] == 1 or vec.shape[1] == 1
    return vec.ravel()

def _scal(mat):
    c = mat.flatten()
    assert len(c) == 1
    return c[0]

def delta_Rt( Rt0, Rt1):

    """
    motion from camera0 to camera1
    
    [ R | t ]  [ R0 | t0 ]  _  [ R1 | t1 ]
    [ 0 | 1 ]  [  0 |  1 ]  -  [  0 |  1 ]  

    R = R1*R0.T
    t = t1 - R*t0
    """

    R0, t0 = Rt0
    R1, t1 = Rt1

    R = np.dot(R1, R0.T)
    t = t1 - np.dot(R, t0)

    return R, t

def transform_plane(normal, distance, R, t):
    """
    Assumption:
    3d plane is parameterized as:
        nx + d = 0
        d > 0
    this means n is pointing toward origin.
    """

    new_normal = np.dot(R, _col(normal))
    # new_distance = np.dot( np.dot(_row(normal), R.T), _col(t) ) + distance
    new_distance = -np.dot( np.dot(_row(normal), R.T), _col(t) ) + distance
    if _scal(new_distance) < 0:
        new_normal = -new_normal
        new_distance = -new_distance

    return _vec(new_normal), _scal(new_distance)

def warp_from_camera_motion(R0, t0, R1, t1, normal, distance, K1, K0_inv=None):

    """
    R0, t0: source camera pose in object frame.
    R1, t1: target camera pose in object frame.
    normal, distance: normal and distance of object's principal plane (in object frame.)
    K1:     target camera's intrinsics
    K0_inv: inverse of source camera's intrisnics

    """

    if K0_inv is None:
        # K0 == K1 
        K0_inv = np.linalg.inv(K1)

    # principal plane in camera 0:
    n, d = transform_plane(normal, distance, R0, t0)

    # motion from camera 0 to camera 1:
    R,t = delta_Rt( (R0,t0), (R1,t1) )

    # euclidean homography:
    H_euc = -R + np.outer(t, n) / d
    # H_euc = R + np.outer(t, n) / d

    # projective homography:
    H = np.dot( np.dot(K1, H_euc), K0_inv)

    return H

