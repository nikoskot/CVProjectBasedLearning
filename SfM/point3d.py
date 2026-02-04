class Point3d:
    
    def __init__(self, coords3d, idx):
        self.coords3d = coords3d
        self.idx = idx
        self.observedAt = [] # [(frame_idx, keypoint_idx)]