import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.core.pose import Pose

class State():
    
    def __init__(self):
        
        self.currentCameraPose = None
        self.trajectory = []    # list of poses
        self.map_points = []
    
    def changeCurrentPose(self, pose):
        self.currentCameraPose = pose
        self.trajectory.append(pose)
        
    def plotTrajectory(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        x = []
        y = []
        z = []
        for p in self.trajectory:
            x.append(p.t_wc[0])
            y.append(p.t_wc[1])
            z.append(p.t_wc[2])

        ax.plot(x, y, z, label='Trajectory')
        # ax.scatter(x, y, z)  # show points

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        plt.show()