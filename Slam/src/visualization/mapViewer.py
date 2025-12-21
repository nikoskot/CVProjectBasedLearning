import rerun as rr
from src.core.state import State

def plotPointCloud(state : State):
    
    rr.log("world/pointCloud/points", rr.Points3D(state.map_points, colors=[255, 255, 255, 255]))