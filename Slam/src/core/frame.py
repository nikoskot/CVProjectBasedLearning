import numpy as np

class Frame:
    
    def __init__(self, image, idx):
        self.image = image
        self.idx = idx
        self.pose = (np.eye(3), np.zeros(3))