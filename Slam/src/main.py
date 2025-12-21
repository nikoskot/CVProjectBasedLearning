import configargparse
import yaml
import rerun as rr
from pathlib import Path
import numpy as np
import cv2 as cv
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.core.camera import Camera, loadIntrinsicsFromJson
from src.core.pose import Pose
from src.core.state import State
from src.io.dataset import Dataset, GroundtruthPosesDataset
from src.visualization.trajectory import plotTrajectory
from src.visualization.mapViewer import plotPointCloud

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
    
    rr.init("SLAM", spawn=True)
    
    dataset = Dataset(args.datasetPath)
    
    ## Dummy data -----------------------------------
    # gtPosesDataset = GroundtruthPosesDataset("D:\\Documents\\Repos\\CVProjectBasedLearning\\Slam\\data\\rgbd_dataset_freiburg1_xyz\\groundtruth.txt", dataset)
    
    # intrinsics = loadIntrinsicsFromJson(args.cameraParamsFile)
    # camera = Camera(intrinsics, 640, 480)
    
    # state = State()
    
    # for frame in dataset.frames:
    #     # frame.pose = Pose(frame.idx, np.eye(3), np.zeros(3) + np.array([0, 0, frame.idx]))
    #     # frame.pose = Pose(frame.idx, np.array([[1, 0, 0], [0, 0.707, -0.707], [0, 0.707, 0.707]]), np.zeros(3) + np.array([0, 0, frame.idx])) # 45 degree around x
    #     frame.pose = gtPosesDataset.gtPoses[frame.idx] # from ground truth poses
        
    #     state.changeCurrentPose(frame.pose)
    
    # plotTrajectory(state, camera, dataset)
    
    # # add some points to the point cloud. Left, right, above and below the final pose
    # state.map_points.append(state.currentCameraPose.t_wc + np.array([-1, 0, 0]))
    # state.map_points.append(state.currentCameraPose.t_wc + np.array([1, 0, 0]))
    # state.map_points.append(state.currentCameraPose.t_wc + np.array([0, -1, 0]))
    # state.map_points.append(state.currentCameraPose.t_wc + np.array([0, 1, 0]))
    # plotPointCloud(state)
    # ----------------------------------------
    
    intrinsics = loadIntrinsicsFromJson(args.cameraParamsFile)
    camera = Camera(intrinsics, 640, 480)
    state = State()
    orb = cv.ORB_create()
    
    # t = 0
    currentPose = Pose(dataset.frames[0].idx, np.eye(3), np.zeros(3))
    dataset.frames[0].pose = currentPose
    state.changeCurrentPose(currentPose)
    
    # t >= 1
    for t in range(1, len(dataset.frames)):
        print(f"Current frame {t}, previous frame {t-1}.")
        rr.set_time("frameId", sequence=dataset.frames[t].idx)
        
        currentFrame = dataset.frames[t]
        previousFrame = dataset.frames[t-1]
        
        # Find the keypoints of the previous frame with ORB
        keypointsPreviousAll = orb.detect(previousFrame.image, None)
        # Compute the descriptors of the previous frame with ORB
        keypointsPreviousAll, descriptorsPreviousAll = orb.compute(previousFrame.image, keypointsPreviousAll)
        # Draw all the keypoints of the previous frame location,not size and orientation
        previousAnnotated = cv.drawKeypoints(previousFrame.image.copy(), keypointsPreviousAll, None, color=(0,255,0), flags=0)
        # print(f"Detected {len(keypointsPreviousAll)} keypoints in frame {t-1}.")
        rr.log("featureCorrespondence/keypointsNumber/previousFrameAll", rr.Scalars(len(keypointsPreviousAll)))
        
        # Find the keypoints of the current frame with ORB
        keypointsCurrentAll = orb.detect(currentFrame.image, None)
        # Compute the descriptors of the current frame with ORB
        keypointsCurrentAll, descriptorsCurrentAll = orb.compute(currentFrame.image, keypointsCurrentAll)
        # Draw all the keypoints of the previous frame location,not size and orientation
        currentAnnotated = cv.drawKeypoints(currentFrame.image.copy(), keypointsCurrentAll, None, color=(0,255,0), flags=0)
        # print(f"Detected {len(keypointsCurrentAll)} keypoints in frame {t}.")
        rr.log("featureCorrespondence/keypointsNumber/currentFrameAll", rr.Scalars(len(keypointsCurrentAll)))
        
        # Feature correspondence/matching
        matchedPixelsPreviousFrame = []
        matchedPixelsCurrentFrame = []
        # Create BFMatcher object
        bfMatcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bfMatcher.match(descriptorsPreviousAll, descriptorsCurrentAll)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        # Draw first 20 matches.
        framesAnnotatedWithMatches = cv.drawMatches(previousFrame.image.copy(), keypointsPreviousAll, currentFrame.image.copy(), keypointsCurrentAll, matches[:20], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        rr.log("featureCorrespondence/annotatedMatches", rr.Image(framesAnnotatedWithMatches))
    
        # Essential matrix calculation
        # Get the correspondences in pixel coordinates for the 100 best matches
        for m in matches:
            px1 = keypointsPreviousAll[m.queryIdx].pt
            px2 = keypointsCurrentAll[m.trainIdx].pt
            matchedPixelsPreviousFrame.append(px1)
            matchedPixelsCurrentFrame.append(px2)
        
        essentialMatrix, inliersMask = cv.findEssentialMat(np.array(matchedPixelsPreviousFrame), np.array(matchedPixelsCurrentFrame), camera.K, method=cv.RANSAC)
        inliersNumber = np.sum(inliersMask)
        outliersNumber = len(inliersMask) - inliersNumber
        rr.log("essenstialMatricCalculation/mathesClassification/inliers", rr.Scalars(inliersNumber))
        rr.log("essenstialMatricCalculation/mathesClassification/outliers", rr.Scalars(outliersNumber))
        
        # If not enough inliers, use previous pose and move to next frame pair
        if (inliersNumber < 30) or (inliersNumber / len(inliersMask) < 0.3):
            print("Not enough inliers for pose estimation. Skipping frame. Using previous pose.")
            
            currentPose = dataset.frames[t-1].pose
            dataset.frames[t].pose = currentPose
            state.changeCurrentPose(currentPose)
            
            continue
        
        # Else, calculate new pose
        retval, Rrelative, trelative, inliersMask = cv.recoverPose(E=essentialMatrix, points1=np.array(matchedPixelsPreviousFrame), points2=np.array(matchedPixelsCurrentFrame), cameraMatrix=camera.K, mask=inliersMask)
        # retval, essentialMatrix, Rrelative, trelative, inliersMask = cv.recoverPose(points1=np.array(matchedPixelsPreviousFrame), points2=np.array(matchedPixelsCurrentFrame), cameraMatrix1=camera.K, distCoeffs1=0, cameraMatrix2=camera.K, distCoeffs2=0, mask=inliersMask)
        # trelative = trelative / np.linalg.norm(trelative)
        # t_prev_unit = dataset.frames[t-1].pose.t_wc
        # if np.dot(trelative.flatten(), t_prev_unit) < 0:
        #     trelative *= -1
        # Rrelative = np.eye(3)
        # trelative = np.array([[0, 0, 1]]).T
        Rnew = previousFrame.pose.R_wc @ Rrelative
        tnew = previousFrame.pose.t_wc + (previousFrame.pose.R_wc @ trelative).squeeze()
        
        currentPose = Pose(dataset.frames[t].idx, Rnew, tnew)
        dataset.frames[t].pose = currentPose
        state.changeCurrentPose(currentPose)
        
    plotTrajectory(state, camera, dataset)    

        
if __name__ == "__main__":
    main()