import numpy as np
from numpy.typing import NDArray


class MapPoint(): # AKA Landmark
    
    def __init__(self, id : int, position : NDArray[np.float32]):
        
        self.id = id
        self.position = position
        self.observations = []  # [(frame_id, keypoint_idx)]
        
    def addObservation(self, frameId : int, keypointidx : int):
        self.observations.append([frameId, keypointidx])