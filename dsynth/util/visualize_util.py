import numpy as np

def visualize_mask(I, M, bg_val=1.):
    I = np.float32(I)/255.
    M = np.float32(M)/255.
    M = np.stack([M,M,M], axis=-1)

    B = np.ones(I.shape, np.float32) * bg_val

    V = M*I + (1.-M)*B

    return V
