class Frame:
    
    def __init__(self, image, idx):
        self.image = image
        self.idx = idx
        self.keypoints = []
        self.descriptors = []
        self.observations = {} # keypoint_idx: 3dPoint_idx
        
    def __init__(self, image, idx, keypoints, descriptors):
        self.image = image
        self.idx = idx
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.observations = {} # keypoint_idx: 3dPoint_idx
