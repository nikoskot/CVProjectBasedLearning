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
    parser = configargparse.ArgParser(default_config_files=["VisualOdometry\\visualOdometryConfig.yaml"])
    parser.add("--configFile", is_config_file=True, help='config file path')
    # parser = argparse.ArgumentParser(description="Camera Calibration")
    parser.add("--dataPath", type=lambda p: Path(p).resolve(), default="VisualOdometry\\data")
    parser.add("--gtPosesPath", type=lambda p: Path(p).resolve(), default="VisualOdometry\\data\\poses.txt")
    parser.add("--resultsSavePath", type=lambda p: Path(p).resolve(), default="VisualOdometry\\visualOdometryResults")
    parser.add("--cameraParamsFile", type=lambda p: Path(p).resolve(), default="VisualOdometry\\data\\intrinsics.json")
    return parser

def saveArgsToYaml(args, filename):
    # Convert Namespace to dict
    args_dict = vars(args)
    # Dump to YAML file
    with open(filename, 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False)

def loadCameraParams(filePath):
    try:
        with open(Path.joinpath(filePath), "r") as f:
            data = json.load(f)
            
        params = {
            "leftCameraMatrix": np.array(data["leftCameraMatrix"], dtype=np.float64),
            "leftDistortionCoeffs": np.array(data["leftDistortionCoeffs"], dtype=np.float64),
            "rightCameraMatrix": np.array(data["rightCameraMatrix"], dtype=np.float64),
            "rightDistortionCoeffs": np.array(data["rightDistortionCoeffs"], dtype=np.float64),
            "rmse": data["rmse"],
            "R": np.array(data["R"], dtype=np.float64),
            "T": np.array(data["T"], dtype=np.float64),
            "E": np.array(data["E"], dtype=np.float64),
            "F": np.array(data["F"], dtype=np.float64),
        }
        
        return params
    
    except Exception as e:
        print(f"Could not load calibration results from file {filePath}. \n Exception {e}")

def loadCameraMatrices(filepath):
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

def loadImages(folderPath):
    imagePaths = [os.path.join(folderPath, file) for file in sorted(os.listdir(folderPath))]
    return [cv.imread(path, cv.IMREAD_GRAYSCALE) for path in imagePaths]

def loadPoses(filepath):
    poses = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            T = np.fromstring(line, dtype=np.float64, sep=' ')
            T = T.reshape(3, 4)
            T = np.vstack((T, [0, 0, 0, 1]))
            poses.append(T)
    return poses

def formTransformation(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

def getMatches(t, images, orb, featuresMatcher):

    # detect and compute ORB keypoints and descriptors for previous and current images
    kp1, des1 = orb.detectAndCompute(images[t-1], None)
    kp2, des2 = orb.detectAndCompute(images[t], None)

    matches = featuresMatcher.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            good.append([m])

    matches1 = np.float32([ kp1[m[0].queryIdx].pt for m in good ])
    matches2 = np.float32([ kp2[m[0].trainIdx].pt for m in good ])

    rr.log("/matches", rr.Image(cv.drawMatchesKnn(images[t-1], kp1, images[t], kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)))
    
    return matches1, matches2

def decompEssentialMat(E, q1, q2, K, P):
    R1, R2, t = cv.decomposeEssentialMat(E)
    T1 = formTransformation(R1,np.ndarray.flatten(t))
    T2 = formTransformation(R2,np.ndarray.flatten(t))
    T3 = formTransformation(R1,np.ndarray.flatten(-t))
    T4 = formTransformation(R2,np.ndarray.flatten(-t))
    transformations = [T1, T2, T3, T4]
    
    # Homogenize K
    K = np.concatenate((K, np.zeros((3,1))), axis = 1)

    # List of projections
    projections = [K @ T1, K @ T2, K @ T3, K @ T4]

    np.set_printoptions(suppress=True)

    # print ("\nTransform 1\n" +  str(T1))
    # print ("\nTransform 2\n" +  str(T2))
    # print ("\nTransform 3\n" +  str(T3))
    # print ("\nTransform 4\n" +  str(T4))

    positives = []
    for Proj, T in zip(projections, transformations):
        hom_Q1 = cv.triangulatePoints(P, Proj, q1.T, q2.T)
        hom_Q2 = T @ hom_Q1
        # Un-homogenize
        Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
        Q2 = hom_Q2[:3, :] / hom_Q2[3, :]  

        total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
        relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1)/
                                    np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
        positives.append(total_sum + relative_scale)
        

    # Decompose the Essential matrix using built in OpenCV function
    # Form the 4 possible transformation matrix T from R1, R2, and t
    # Create projection matrix using each T, and triangulate points hom_Q1
    # Transform hom_Q1 to second camera using T to create hom_Q2
    # Count how many points in hom_Q1 and hom_Q2 with positive z value
    # Return R and t pair which resulted in the most points with positive z

    max = np.argmax(positives)
    if (max == 2):
        # print(-t)
        return R1, np.ndarray.flatten(-t)
    elif (max == 3):
        # print(-t)
        return R2, np.ndarray.flatten(-t)
    elif (max == 0):
        # print(t)
        return R1, np.ndarray.flatten(t)
    elif (max == 1):
        # print(t)
        return R2, np.ndarray.flatten(t)
        
def getPose(matchesPrev, matchesCurr, K, P):
    # Estimate Essential matrix
    E, mask = cv.findEssentialMat(matchesPrev, matchesCurr, K)

    # Recover pose from Essential matrix
    # _, R, t, mask = cv.recoverPose(E, matchesPrev, matchesCurr)
    R, t = decompEssentialMat(E, matchesPrev, matchesCurr, K, P)

    # Form the transformation matrix
    transform = formTransformation(R, t.flatten())

    return transform

def visualOdometry(images, gtPoses, K, P):
    # Initialize detector and matcher
    orb = cv.ORB_create(3000)
    featuresMatcher = cv.BFMatcher()
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    featuresMatcher = cv.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    rr.log("/camera", rr.Pinhole(resolution=[images[0].shape[1], images[0].shape[0]], focal_length=[K[0,0], K[1,1]], principal_point=[K[0, 2], K[1, 2]]), static=True)

    gtTrajectory = []
    estimatedTrajectory = []
    for t in tqdm.tqdm(range(len(gtPoses))):
        rr.set_time("frameId", sequence=t)

        rr.log("/camera", rr.Image(images[t]))

        if t == 0:
            currentPose = gtPoses[t]
        else:
            print(f"Processing current frame {t}, previous frame {t-1}.")

            matchesPrev, matchesCurr = getMatches(t, images, orb, featuresMatcher)
            transform = getPose(matchesPrev, matchesCurr, K, P)
            currentPose = np.matmul(currentPose, np.linalg.inv(transform))

        gtTrajectory.append((gtPoses[t][0, 3], gtPoses[t][1, 3], gtPoses[t][2, 3]))
        estimatedTrajectory.append((currentPose[0, 3], currentPose[1, 3], currentPose[2, 3]))
        rr.log("/camera", rr.Transform3D(mat3x3=currentPose[:3,:3], translation=currentPose[:3,3]))

        rr.log("/groundTruthTrajectory", rr.LineStrips3D([gtTrajectory]))
        rr.log("/estimatedTrajectory", rr.LineStrips3D([estimatedTrajectory]))
        rr.log("/error", rr.Scalars(np.linalg.norm(np.array(gtTrajectory[-1]) - np.array(estimatedTrajectory[-1]))))

    pass

def main():
    pareser = getParser()
    args = pareser.parse_args()

     # Create necessary folders/paths
    print("---Creating path for visual odometry results.---")
    args.resultsSavePath = Path.joinpath(args.resultsSavePath, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(args.resultsSavePath, exist_ok=True)
    # Save arguments use to file
    saveArgsToYaml(args, Path.joinpath(args.resultsSavePath, "config.yaml"))
    
    print("---Starting visual odometry with tracking with:")
    print('\n'.join(f"{k}: {v}" for k, v in vars(args).items()))
    
    # print(f"---Load camera parameters from {args.cameraParamsFile}.---")
    # cameraParams = loadCameraParams(args.cameraParamsFile)
    # print(f"Loaded calibration parameters: \n {cameraParams}")

    print(f"---Load camera matrices from {args.cameraParamsFile}.---")
    K, P = loadCameraMatrices(args.cameraParamsFile)
    print(f"Loaded camera matrices: \n K = {K} \n P = {P}")

    print(f"---Loading images from {args.dataPath}.---")
    images = loadImages(args.dataPath)
    print(f"Loaded {len(images)} images.")
    
    print(f"---Loading ground truth poses from {args.gtPosesPath}.---")
    gtPoses = loadPoses(args.gtPosesPath)
    print(f"Loaded {len(gtPoses)} ground truth poses.")

    rr.init("Visual Odometry", spawn=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    visualOdometry(images, gtPoses, K, P)

if __name__ == '__main__':
    main()