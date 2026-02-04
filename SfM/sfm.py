import configargparse
import yaml
import json
import rerun as rr
from pathlib import Path
import numpy as np
import cv2 as cv
import sys
import os
import datetime
import tqdm
sys.path.append(str(Path(__file__).resolve().parent.parent))


def getParser():
    parser = configargparse.ArgParser(default_config_files=["SfM\\sfmConfig.yaml"])
    parser.add("--configFile", is_config_file=True, help='config file path')
    # parser = argparse.ArgumentParser(description="Camera Calibration")
    parser.add("--dataPath", type=lambda p: Path(p).resolve(), default="SfM\\data\\images")
    # parser.add("--gtPosesPath", type=lambda p: Path(p).resolve(), default="SfM\\data\\poses.txt")
    parser.add("--resultsSavePath", type=lambda p: Path(p).resolve(), default="SfM\\sfmResults")
    parser.add("--cameraParamsFile", type=lambda p: Path(p).resolve(), default="SfM\\data\\intrinsics.json")
    return parser

def saveArgsToYaml(args, filename):
    # Convert Namespace to dict
    args_dict = vars(args)
    # Dump to YAML file
    with open(filename, 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False)
        
def loadIntrinsicsFromJson(jsonFilePath : str):
    intrinsics = {}
    K, P = None, None
    
    try:
        with open(jsonFilePath, 'r') as file:
            data = json.load(file)
            intrinsics['fx'] = data['fx']
            intrinsics['fy'] = data['fy']
            intrinsics['cx'] = data['cx']
            intrinsics['cy'] = data['cy']
            
            P = np.zeros((3, 4))
            P[0, 0] = intrinsics['fx']
            P[1, 1] = intrinsics['fy']
            P[0, 2] = intrinsics['cx']
            P[1, 2] = intrinsics['cy']
            P[2, 2] = 1.0
            K = P[0:3, 0:3]
            
    except Exception as e:
        print(f"Could not load calibration results from file jsonFilePath. \n Exception {e}")
        
    return intrinsics, K, P

def loadImages(folderPath):
    imagePaths = [os.path.join(folderPath, file) for file in sorted(os.listdir(folderPath))]
    return [cv.imread(path, cv.IMREAD_GRAYSCALE) for path in imagePaths]

def sfm(images, K, P):
    print("OK")
    
def main():
    pareser = getParser()
    args = pareser.parse_args()

     # Create necessary folders/paths
    print("---Creating path for SfM results.---")
    args.resultsSavePath = Path.joinpath(args.resultsSavePath, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(args.resultsSavePath, exist_ok=True)
    # Save arguments use to file
    saveArgsToYaml(args, Path.joinpath(args.resultsSavePath, "config.yaml"))
    
    print("---Starting SfM with tracking with:")
    print('\n'.join(f"{k}: {v}" for k, v in vars(args).items()))

    print(f"---Load camera intrinsics from {args.cameraParamsFile}.---")
    intrinsics, K, P = loadIntrinsicsFromJson(args.cameraParamsFile)
    print(f"Loaded camera intrinsics: \n K = {K} \n P = {P}")

    print(f"---Loading images from {args.dataPath}.---")
    images = loadImages(args.dataPath)
    print(f"Loaded {len(images)} images.")

    rr.init("SfM", spawn=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    sfm(images, K, P)

if __name__ == '__main__':
    main()