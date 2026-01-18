import rerun as rr
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.core.camera import Camera
from src.core.pose import Pose
from src.core.state import State
from src.io.dataset import Dataset

def plotTrajectory(state : State, camera : Camera, dataset : Dataset):
    
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    
    positions = []
    
    rr.log("world/trajectory/camera", rr.Pinhole(resolution=[camera.width, camera.height], focal_length=[camera.fx, camera.fy], principal_point=[camera.cx, camera.cy]), static=True)
    
    for pose in state.trajectory:
        
        poseCopy = Pose(pose.frameId, pose.R_cw.T, -pose.R_cw.T @ pose.t_cw)
        # poseCopy = Pose(pose.frameId, pose.R_cw, pose.t_cw)
        
        position = np.array([poseCopy.t_cw[0], poseCopy.t_cw[1], poseCopy.t_cw[2]])
        positions.append([poseCopy.t_cw[0], poseCopy.t_cw[1], poseCopy.t_cw[2]])
        
        rr.set_time("frameId", sequence=poseCopy.frameId)
        rr.log("world/trajectory/cameraCenter", rr.Points3D(position, radii=0.5))
        
        rr.log("world/trajectory/camera", rr.Transform3D(mat3x3=poseCopy.R_cw, translation=poseCopy.t_cw))
        rr.log("world/trajectory/camera", rr.Image(dataset.frames[poseCopy.frameId].image))
    
        rr.log("world/trajectory/line", rr.LineStrips3D([positions]))
