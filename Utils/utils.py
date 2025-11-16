import numpy as np
from scipy.spatial.transform import Rotation

def _to_rotmat(r):
        r = np.asarray(r)
        if r.shape == (3,) or r.shape == (3,1) or r.shape == (1,3):
            return Rotation.from_rotvec(r.reshape(3,)).as_matrix()
        if r.shape == (3,3):
            return r
        raise ValueError(f"Unsupported rotation shape: {r.shape}")

def _to_vec(t):
        return np.asarray(t).reshape(3,)