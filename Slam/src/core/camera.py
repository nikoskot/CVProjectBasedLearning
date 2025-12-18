import numpy as np
import json

class Camera():
    
    def __init__(self, intrinsics, width=1920, height=1080):
        self.fx = intrinsics['fx']
        self.fy = intrinsics['fy']
        self.cx = intrinsics['cx']
        self.cy = intrinsics['cy']
        
        self.K = np.zeros((3, 3))
        self.K[0, 0] = self.fx
        self.K[0, 2] = self.cx
        self.K[1, 1] = self.fy
        self.K[1, 2] = self.cy
        self.K[2, 2] = 1
        
        self.distortion = [0, 0, 0, 0, 0]
        
        self.width = width
        self.height = height
        
    
    def project3dTo2d(self, X_world, R, t):
        """
        xWorld: (3,) world point [X, Y, Z]
        intrinsics: (3,3) intrinsic matrix
        R: (3,3) rotation matrix
        t: (3,1) translation vector
        """
        if t.shape == (3,):
            t = t.reshape(3,1)

        # Convert to homogeneous world point (4x1)
        X_h = np.append(X_world, 1)  # (4,)

        # Projection matrix (3x4)
        P = self.K @ np.hstack((R, t))    # [R|t]

        # Project
        x_h = P @ X_h                # (3,)

        # Dehomogenize
        u = x_h[0] / x_h[2]
        v = x_h[1] / x_h[2]

        return np.array([u, v])
    
    def backProjection(self, u, v):
        """
        Returns a unit ray direction in the camera frame.
        Shape: (3,)
        """
        
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        
        ray = np.array([x, y, 1])
        
        return ray / np.linalg.norm(ray)


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
        
    return intrinsics

if __name__ == "__main__":
    intrinsics = loadIntrinsicsFromJson("D:\\Documents\\Repos\\CVProjectBasedLearning\\Slam\\data\\rgbd_dataset_freiburg1_xyz\\intrinsics.json")
    camera = Camera(intrinsics)
    print(camera.K)
    print(camera.distortion)
    
    print(f"Back projection of center pixel = {camera.backProjection(camera.cx, camera.cy)}")
    print(f"Back projection of pixel (cx + fx, cy) = {camera.backProjection(camera.cx + camera.fx, camera.cy)}")
    
    print(f"Projection of point (0, 0, 1) = {camera.project3dTo2d(np.array([0, 0, 1]), np.eye(3), np.zeros(3))}")