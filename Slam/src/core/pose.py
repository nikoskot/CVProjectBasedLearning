import numpy as np
from numpy.typing import NDArray

class Pose():
    '''
    R rotates from camera → world
    t is the camera origin in world coordinates
    '''
    
    def __init__(self, frameId : int, R_wc :NDArray[np.float32], t_wc : NDArray[np.float32]):
        
        self.frameId = frameId
        self.R_wc = R_wc
        self.t_wc = t_wc
        
        
        