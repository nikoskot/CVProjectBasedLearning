import numpy as np
import json

class Camera():
    
    def __init__(self, intrinsics):
        self.fx = intrinsics['fx']
        self.fy = intrinsics['fy']
        self.cx = intrinsics['cx']
        self.cy = intrinsics['cy']
        
        self.projectionMatrix = np.zeros((3, 3))
        self.projectionMatrix[0, 0] = self.fx
        self.projectionMatrix[0, 2] = self.cx
        self.projectionMatrix[1, 1] = self.fy
        self.projectionMatrix[1, 2] = self.cy
        self.projectionMatrix[2, 2] = 1
        
        self.distortion = [0, 0, 0, 0, 0]
    
    def project3dTo2d(self, worldCoords):
        return self.projectionMatrix @ worldCoords
    
    def backProjection(self, pixelCoords):
        print("Back projection not implemented.")


def loadIntrinsicsFromJson(jsonFilePath):
    intrinsics = {}
    
    try:
        with open(jsonFilePath, 'r') as file:
            data = json.load(file)
            intrinsics['fx'] = data['fx']
            intrinsics['fy'] = data['fy']
            intrinsics['cx'] = data['cx']
            intrinsics['cy'] = data['cy']
            
    except Exception as e:
        print(f"Could not load calibration results from file jsonFilePath. \n Exception {e}")