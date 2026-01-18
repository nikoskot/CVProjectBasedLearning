import numpy as np
from numpy.typing import NDArray

class Frame:
    
    def __init__(self, image : NDArray[np.float32], idx : int, timestamp : float):
        self.image = image
        self.idx = idx
        self.timestamp = timestamp
        self.pose = (np.eye(3), np.zeros(3))
        self.keypoints = []
        self.descriptors = []
        self.observations = [] # [(landmark_id, keypoint_idx)]
        self.trajectoryIndex = -1