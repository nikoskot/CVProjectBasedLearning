class Pose():
    '''
    R rotates from camera → world
    t is the camera origin in world coordinates
    '''
    
    def __init__(self, frameId, R_wc, t_wc):
        
        self.frameId = frameId
        self.R_wc = R_wc
        self.t_wc = t_wc
        
        
        