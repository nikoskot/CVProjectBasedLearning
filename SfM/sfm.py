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
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parent.parent))
from frame import Frame
from point3d import Point3d


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

def formTransformation(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    
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

def sfm(images, K, P):
    frames = []
    poses = []
    points3d = []   # The 3d points that are created in the scene. List of Point3d objects
    matches = [[None for _ in range(len(images))] for _ in range(len(images))]      # 2D array that contatins the feature matches between all images. It contains tuples of the coordinates of the feature in the first image the coordinates in the second image
    matchesKeypointsIdxs = [[None for _ in range(len(images))] for _ in range(len(images))]     # Same as the above but it contatins the indexes of the keypoints, not their coordinates
    projections = []
     
    rr.log("/camera", rr.Pinhole(resolution=[images[0].shape[1], images[0].shape[0]], focal_length=[K[0,0], K[1,1]], principal_point=[K[0, 2], K[1, 2]]), static=True)
    
    # Calculate keypoints and descriptors for all images
    print("Calculate keypoints and descriptors for all images")
    featureDetector = cv.ORB_create(5000)
    # featureDetector = cv.SIFT_create(nfeatures=5000)
    for imgIdx, img in tqdm(enumerate(images)):
        keypoints, descriptors = featureDetector.detectAndCompute(img, None)
        frame = Frame(img, imgIdx, keypoints, descriptors)
        frames.append(frame)
        rr.set_time("frameId", sequence=imgIdx)
        rr.log("/camera", rr.Image(img))
    
    # Match features between all images
    print("Match features between all images")
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    featuresMatcher = cv.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
    
    for i in tqdm(range(len(frames))):
        for j in range(i, len(frames)):
            if i == j:
                matches[i][j] = ()
            else:
                # Find matches between the two images
                currentMatches = featuresMatcher.knnMatch(frames[i].descriptors, frames[j].descriptors, k=2)
                # Apply ratio test
                good = []
                for m,n in currentMatches:
                    if m.distance < 0.5*n.distance:
                        good.append(m)
                
                # Save the coordinates of the matches
                matches1 = np.float32([ frames[i].keypoints[m.queryIdx].pt for m in good ])
                matches2 = np.float32([ frames[j].keypoints[m.trainIdx].pt for m in good ])
                matches[i][j] = (matches1, matches2)
                matches[j][i] = (matches2, matches1)
                
                # Save the indexes of the keypoints of the matches
                matchesKeypointsIdxs1 = np.int32([ m.queryIdx for m in good ])
                matchesKeypointsIdxs2 = np.int32([ m.trainIdx for m in good ])
                matchesKeypointsIdxs[i][j] = (matchesKeypointsIdxs1, matchesKeypointsIdxs2)
                matchesKeypointsIdxs[j][i] = (matchesKeypointsIdxs2, matchesKeypointsIdxs1)
    
    # Pose of the 1st image
    poses.append(formTransformation(np.eye(3), np.zeros(3)))
    rr.set_time("frameId", sequence=0)
    # Rerun need camera -> world coordinate system transformation, so we need to invert the pose
    rr.log("/camera", rr.Transform3D(mat3x3=np.linalg.inv(poses[0])[:3,:3], translation=np.linalg.inv(poses[0])[:3,3]))
    rr.log("/annotatedImages", rr.Image(cv.drawKeypoints(images[0], frames[0].keypoints, 0, (255, 0, 0), flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)))
    
    # Projection matrix of the 1st image
    projections.append(P)
    
    # Estimate Essential matrix for 1st and 2nd frame
    E, mask = cv.findEssentialMat(matches[0][1][0], matches[0][1][1], K)
    # Recover relative pose from Essential matrix
    R, t = decompEssentialMat(E, matches[0][1][0], matches[0][1][1], K, P)
    # Form the transformation matrix. This is world -> camera coordinate system
    relativePoseTransform = formTransformation(R, t.flatten())
    # Get the pose of the 2nd camera
    # poses.append(np.matmul(poses[0], np.linalg.inv(relativePoseTransform)))
    poses.append(relativePoseTransform @ poses[0]) 
    rr.set_time("frameId", sequence=1)
    # Rerun need camera -> world coordinate system transformation, so we need to invert the pose
    rr.log("/camera", rr.Transform3D(mat3x3=np.linalg.inv(poses[1])[:3,:3], translation=np.linalg.inv(poses[1])[:3,3]))
    rr.log("/annotatedImages", rr.Image(cv.drawKeypoints(images[1], frames[1].keypoints, 0, (255, 0, 0), flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)))
    # Projection matrix of the 2nd image
    projections.append(np.concatenate((K, np.zeros((3,1))), axis = 1) @ poses[1])
    
    # Get the 3D points from the first two images
    X_h = cv.triangulatePoints(projections[0], projections[1], matches[0][1][0].T, matches[0][1][1].T)
    X = X_h[:3, :] / X_h[3, :]
    for i in range(X.shape[1]):
        # Create a point3d object for the 3d point
        point3dIdx = len(points3d) + 1
        point3d = Point3d(X[:, i], point3dIdx)
        # Add the id of the frames and the keypoints that the 3d point is observed at
        point3d.observedAt.append((0, matchesKeypointsIdxs[0][1][0][i]))
        point3d.observedAt.append((1, matchesKeypointsIdxs[0][1][1][i]))
        # Add the id of the keypoint and the 3d point at the frames' observations
        frames[0].observations.append((matchesKeypointsIdxs[0][1][0][i], point3dIdx))
        frames[1].observations.append((matchesKeypointsIdxs[0][1][1][i], point3dIdx))
        
        points3d.append(point3d)
    
    rr.log("/pointCloud/points", rr.Points3D([p.coords3d for p in points3d], colors=[255, 255, 255, 255], radii=0.01), static=True)
    
    pass
    
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