import numpy as np

class Frame:
    
    def __init__(self, image, idx, timestamp):
        self.image = image
        self.idx = idx
        self.timestamp = timestamp
        self.pose = (np.eye(3), np.zeros(3))