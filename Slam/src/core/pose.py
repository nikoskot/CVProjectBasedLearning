import numpy as np
from numpy.typing import NDArray

class Pose():
    '''
    R_cw: from camera to world
    t is the camera origin in world coordinates
    '''
    
    def __init__(self, frameId : int, R_cw :NDArray[np.float32], t_cw : NDArray[np.float32]):
        
        self.frameId = frameId
        self.R_cw = R_cw
        self.t_cw = t_cw
        
        
        