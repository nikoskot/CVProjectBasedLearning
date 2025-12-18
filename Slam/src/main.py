import configargparse
import yaml
from pathlib import Path
import numpy as np
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.core.camera import Camera, loadIntrinsicsFromJson
from src.core.pose import Pose
from src.core.state import State
from src.io.dataset import Dataset, GroundtruthPosesDataset
from src.visualization.trajectory import plotPointCloud, plotTrajectory

def getParser():
    parser = configargparse.ArgParser(default_config_files=["Slam\configs\default.yaml"])
    parser.add("--configFile", is_config_file=True, help='config file path')
    # parser = argparse.ArgumentParser(description="Camera Calibration")
    parser.add("--datasetPath", type=lambda p: Path(p).resolve(), default="Slam\data")
    parser.add("--resultsSavePath", type=lambda p: Path(p).resolve(), default="Slam\results")
    parser.add("--cameraParamsFile", type=lambda p: Path(p).resolve(), default="Slam\data\camera\intrinsics.json")
    return parser

def main():
    parser = getParser()
    args = parser.parse_args()
    print(f"---Starting SLAM process with: \n {vars(args)}---")
    
    dataset = Dataset(args.datasetPath)
    
    gtPosesDataset = GroundtruthPosesDataset("D:\\Documents\\Repos\\CVProjectBasedLearning\\Slam\\data\\rgbd_dataset_freiburg1_xyz\\groundtruth.txt", dataset)
    
    intrinsics = loadIntrinsicsFromJson(args.cameraParamsFile)
    camera = Camera(intrinsics, 640, 480)
    
    state = State()
    
    for frame in dataset.frames:
        frame.pose = Pose(frame.idx, np.eye(3), np.zeros(3) + np.array([0, 0, frame.idx]))
        frame.pose = Pose(frame.idx, np.array([[1, 0, 0], [0, 0.707, -0.707], [0, 0.707, 0.707]]), np.zeros(3) + np.array([0, 0, frame.idx])) # 45 degree around x
        frame.pose = gtPosesDataset.gtPoses[frame.idx] # from ground truth poses
        
        state.changeCurrentPose(frame.pose)
    
    plotTrajectory(state, camera, dataset)

if __name__ == "__main__":
    main()