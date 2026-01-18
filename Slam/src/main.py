import configargparse
import yaml
import rerun as rr
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import cv2 as cv
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.core.camera import Camera, loadIntrinsicsFromJson
from src.core.pose import Pose
from src.core.state import State
from src.core.mapPoint import MapPoint
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

def index_of_ith_one(arr: list[int], i: int) -> int:
    count = 0
    for idx, value in enumerate(arr):
        if value[0] == 255:
            if count == i:
                return idx
            count += 1
    raise ValueError(f"List contains fewer than {i+1} ones")

def projectionResiduals(x, frameToLandmarkIdToPixel, landmarksIdsUsed, firstPose, K):
    numberOfKeyframes = len(frameToLandmarkIdToPixel)
    residuals = []

    for i in range(numberOfKeyframes):
        rotationMatrix = np.zeros((3,3))
        translation = np.zeros(3)
        if i == 0:
            rotationMatrix = firstPose.R_cw
            translation = firstPose.t_cw.reshape(3,)
        else:
            rotationMatrix = Rotation.from_rotvec([x[(i-1)*6 + 0], x[(i-1)*6 + 1], x[(i-1)*6 + 2]]).as_matrix()
            translation = np.array([x[(i-1)*6 + 3], x[(i-1)*6 + 4], x[(i-1)*6 + 5]]).reshape(3,)
            
        for j in frameToLandmarkIdToPixel.keys():
            for lid, pixel in frameToLandmarkIdToPixel[j].items():

                landmark_W = np.array([x[6*(numberOfKeyframes-1) + landmarksIdsUsed.index(lid)*3 + 0], x[6*(numberOfKeyframes-1) + landmarksIdsUsed.index(lid)*3 + 1], x[6*(numberOfKeyframes-1) + landmarksIdsUsed.index(lid)*3 + 2]]).reshape(3,)
                landmark_C = rotationMatrix.T @ (landmark_W - translation).reshape(3,)

                # if landmark_C[2] <= 0:
                #     continue
                
                # Projection
                u = K[0,0] * landmark_C[0] / landmark_C[2] + K[0,2]
                v = K[1,1] * landmark_C[1] / landmark_C[2] + K[1,2]

                residuals.append(u - pixel[0])
                residuals.append(v - pixel[1])
    
    return residuals


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
    # state.map_points.append(state.currentCameraPose.t_cw + np.array([-1, 0, 0]))
    # state.map_points.append(state.currentCameraPose.t_cw + np.array([1, 0, 0]))
    # state.map_points.append(state.currentCameraPose.t_cw + np.array([0, -1, 0]))
    # state.map_points.append(state.currentCameraPose.t_cw + np.array([0, 1, 0]))
    # plotPointCloud(state)
    # ----------------------------------------
    
    intrinsics = loadIntrinsicsFromJson(args.cameraParamsFile)
    camera = Camera(intrinsics, 640, 480)
    state = State()
    orb = cv.ORB_create()
    
    # t = 0 (is keyframe by default)
    currentPose = Pose(dataset.frames[0].idx, np.eye(3), np.zeros(3))
    dataset.frames[0].pose = currentPose
    state.changeCurrentPose(currentPose)
    state.keyframesIds.append(0)
    dataset.frames[0].trajectoryIndex = 0
    previousFrameT = 0
    landmarkId = 0
    
    # t >= 1
    for t in range(1, len(dataset.frames)):
        print(f"Current frame {t}, previous frame {t-1}.")
        rr.set_time("frameId", sequence=dataset.frames[t].idx)
        
        currentFrame = dataset.frames[t]
        previousFrame = dataset.frames[previousFrameT]
        
        # Find the keypoints of the previous frame with ORB
        keypointsPreviousAll = orb.detect(previousFrame.image, None)
        # Compute the descriptors of the previous frame with ORB
        keypointsPreviousAll, descriptorsPreviousAll = orb.compute(previousFrame.image, keypointsPreviousAll)
        # Draw all the keypoints of the previous frame location,not size and orientation
        previousAnnotated = cv.drawKeypoints(previousFrame.image.copy(), keypointsPreviousAll, None, color=(0,255,0), flags=0)
        # print(f"Detected {len(keypointsPreviousAll)} keypoints in frame {t-1}.")
        rr.log("featureCorrespondence/keypointsNumber/previousFrameAll", rr.Scalars(len(keypointsPreviousAll)))
        # Save keypoints and descriptors of previous frame
        previousFrame.keypoints = keypointsPreviousAll
        previousFrame.descriptors = descriptorsPreviousAll
        
        # Find the keypoints of the current frame with ORB
        keypointsCurrentAll = orb.detect(currentFrame.image, None)
        # Compute the descriptors of the current frame with ORB
        keypointsCurrentAll, descriptorsCurrentAll = orb.compute(currentFrame.image, keypointsCurrentAll)
        # Draw all the keypoints of the previous frame location,not size and orientation
        currentAnnotated = cv.drawKeypoints(currentFrame.image.copy(), keypointsCurrentAll, None, color=(0,255,0), flags=0)
        # print(f"Detected {len(keypointsCurrentAll)} keypoints in frame {t}.")
        rr.log("featureCorrespondence/keypointsNumber/currentFrameAll", rr.Scalars(len(keypointsCurrentAll)))
        # Save keypoints and descriptors of current frame
        currentFrame.keypoints = keypointsCurrentAll
        currentFrame.descriptors = descriptorsCurrentAll
        
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
        framesAnnotatedWithMatches = cv.drawMatches(previousFrame.image.copy(), keypointsPreviousAll, currentFrame.image.copy(), keypointsCurrentAll, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        rr.log("featureCorrespondence/annotatedMatches", rr.Image(framesAnnotatedWithMatches))
        # Histogram of distances
        matchesDistances = [x.distance for x in matches]
        matchesDistancesHistogram = np.histogram(matchesDistances)
        rr.log("featureCorrespondence/distanceHistogram", rr.BarChart(matchesDistancesHistogram[0], abscissa=matchesDistancesHistogram[1]))
    
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
            
            currentPose = dataset.frames[previousFrameT].pose
            dataset.frames[t].pose = currentPose
            state.changeCurrentPose(currentPose)
            dataset.frames[t].trajectoryIndex = len(state.trajectory) - 1
            previousFrameT = t
            continue
        
        # Else, calculate new pose
        # retval, Rrelative, trelative, inliersMask = cv.recoverPose(E=essentialMatrix, points1=np.array(matchedPixelsPreviousFrame), points2=np.array(matchedPixelsCurrentFrame), cameraMatrix=camera.K, mask=inliersMask)
        retval, essentialMatrix, Rrelative, trelative, inliersMask = cv.recoverPose(points1=np.array(matchedPixelsPreviousFrame), points2=np.array(matchedPixelsCurrentFrame), cameraMatrix1=camera.K, distCoeffs1=np.zeros(4), cameraMatrix2=camera.K, distCoeffs2=np.zeros(4))
        # trelative = trelative / np.linalg.norm(trelative)
        # t_prev_unit = dataset.frames[t-1].pose.t_cw
        # if np.dot(trelative.flatten(), t_prev_unit) < 0:
        #     trelative *= -1
        # Rrelative = np.eye(3)
        # trelative = np.array([[0, 0, 1]]).T
        # If Rotatin angleas are above a threshold, skip this frame
        rotationXDegrees = Rotation.from_matrix(Rrelative).as_euler('xyz', degrees=True)[0]
        rotationYDegrees = Rotation.from_matrix(Rrelative).as_euler('xyz', degrees=True)[1]
        rotationZDegrees = Rotation.from_matrix(Rrelative).as_euler('xyz', degrees=True)[2]
        if abs(rotationXDegrees) > 3 or abs(rotationYDegrees) > 3 or abs(rotationZDegrees) > 3:
            print("Frame skipped due to large rotation. Using previous pose.")
            currentPose = dataset.frames[previousFrameT].pose
            dataset.frames[t].pose = currentPose
            state.changeCurrentPose(currentPose)
            dataset.frames[t].trajectoryIndex = len(state.trajectory) - 1
            continue
        
        Rnew = previousFrame.pose.R_cw @ Rrelative
        tnew = previousFrame.pose.t_cw + (previousFrame.pose.R_cw @ trelative).squeeze()
        rr.log("poseEstimation/relativePose/transform", rr.Transform3D(mat3x3=Rrelative, translation=trelative.squeeze()))
        rr.log("poseEstimation/relativePose/scalars/translationMagnitude", rr.Scalars(np.linalg.norm(trelative)))
        rr.log("poseEstimation/relativePose/scalars/rotationXDegrees", rr.Scalars(Rotation.from_matrix(Rrelative).as_euler('xyz', degrees=True)[0]))
        rr.log("poseEstimation/relativePose/scalars/rotationYDegrees", rr.Scalars(Rotation.from_matrix(Rrelative).as_euler('xyz', degrees=True)[1]))
        rr.log("poseEstimation/relativePose/scalars/rotationZDegrees", rr.Scalars(Rotation.from_matrix(Rrelative).as_euler('xyz', degrees=True)[2]))
        
        currentPose = Pose(dataset.frames[t].idx, Rnew, tnew)
        dataset.frames[t].pose = currentPose
        state.changeCurrentPose(currentPose)
        dataset.frames[t].trajectoryIndex = len(state.trajectory) - 1
        previousFrameT = t
        
        # Landmarks and keyframes--------------------
        # Get last saved keyframe
        lastKeyframeId = state.keyframesIds[-1]
        lastKeyframe = dataset.frames[lastKeyframeId]
        
        # Feature correspondence/matching between current frame and last keyframe
        matchedPixelsPreviousKeyframe = []
        matchedPixelsCurrentFrame = []
        
        # Match descriptors.
        matches = bfMatcher.match(lastKeyframe.descriptors, currentFrame.descriptors)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        # Draw first 20 matches.
        lastKeyframeAndCurrentFrameAnnotatedWithMatches = cv.drawMatches(lastKeyframe.image.copy(), lastKeyframe.keypoints, currentFrame.image.copy(), currentFrame.keypoints, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        rr.log("featureCorrespondence/lastKeyframeAndCurrentFrameAnnotatedMatches", rr.Image(lastKeyframeAndCurrentFrameAnnotatedWithMatches))
        # Histogram of distances
        matchesDistances = [x.distance for x in matches]
        matchesDistancesHistogram = np.histogram(matchesDistances)
        rr.log("featureCorrespondence/lastKeyframeAndCurrentFramedistanceHistogram", rr.BarChart(matchesDistancesHistogram[0], abscissa=matchesDistancesHistogram[1]))
        
        # Get the correspondences in pixel coordinates for all the matches
        for m in matches:
            px1 = lastKeyframe.keypoints[m.queryIdx].pt
            px2 = currentFrame.keypoints[m.trainIdx].pt
            matchedPixelsPreviousKeyframe.append(px1)
            matchedPixelsCurrentFrame.append(px2)
        
        # Decide if current frame is keyframe
        translationMagnitudeFromLastKeyframe = np.linalg.norm(currentPose.t_cw - lastKeyframe.pose.t_cw)
        rotationFromLastKeyframe = lastKeyframe.pose.R_cw.T @ currentPose.R_cw
        rot_angle = np.acos((np.trace(rotationFromLastKeyframe) - 1) / 2)
        if (translationMagnitudeFromLastKeyframe <= 2 and rot_angle <= np.pi/18 and len(matches) >= 0.5 * len(lastKeyframe.keypoints)):
            # not a keyframe, skip it
            continue
        print(f"Current frame is a keyframe. t_rel = {translationMagnitudeFromLastKeyframe}, rot_angle = {rot_angle}, matches with last keyframe = {len(matches)}, features of last keyframe = {len(lastKeyframe.keypoints)}")
        
        state.keyframesIds.append(currentFrame.idx)
        
        retval, essentialMatrixLastKeyframeAndCurrentFrame, RrelativeLastKeyframeAndCurrentFrame, trelativeLastKeyframeAndCurrentFrame, inliersMaskLastKeyframeAndCurrentFrame = cv.recoverPose(points1=np.array(matchedPixelsPreviousKeyframe), points2=np.array(matchedPixelsCurrentFrame), cameraMatrix1=camera.K, distCoeffs1=np.zeros(4), cameraMatrix2=camera.K, distCoeffs2=np.zeros(4))
        retval, RrelativeLastKeyframeAndCurrentFrame, trelativeLastKeyframeAndCurrentFrame, inliersMaskLastKeyframeAndCurrentFrame, triangulatedPoints = cv.recoverPose(E=essentialMatrixLastKeyframeAndCurrentFrame, points1=np.array(matchedPixelsPreviousKeyframe), points2=np.array(matchedPixelsCurrentFrame), cameraMatrix=camera.K, distanceThresh=1000)
        triangulatedPoints = triangulatedPoints[:3, :] / triangulatedPoints[3, :]

        # For each new triangulated point
        for triangulatedPointIndex in range(len(triangulatedPoints)):
            keypointIndex = index_of_ith_one(inliersMaskLastKeyframeAndCurrentFrame, triangulatedPointIndex)
            found = False
            new3DPointInWorldCoordinates = (lastKeyframe.pose.R_cw @ np.array(triangulatedPoints[:, triangulatedPointIndex]).reshape(3,)).squeeze() + lastKeyframe.pose.t_cw
            # Search in all the already saved landmarks, if there is already a landmark with the same 3d position
            for lmIndex in range(len(state.map_points)):
                lm = state.map_points[lmIndex]
                # If there is, just add the new observation to the existing landmark
                if lm.position[0] == new3DPointInWorldCoordinates[0] and lm.position[1] == new3DPointInWorldCoordinates[1] and lm.position[2] == new3DPointInWorldCoordinates[2]:
                    lm.addObservation(frameId=currentFrame.idx, keypointidx=keypointIndex)
                    currentFrame.observations.append([lm.id, keypointIndex])
                    found = True
                    break
            # If not found
            if not found:
                newLandmark = MapPoint(landmarkId, new3DPointInWorldCoordinates)
                newLandmark.addObservation(frameId=currentFrame.idx, keypointidx=keypointIndex)
                currentFrame.observations.append([newLandmark.id, keypointIndex])
                state.map_points.append(newLandmark)
                landmarkId += 1
        
        # Local Bundle Adjustment
        baWindowSize = 5
        if len(state.keyframesIds) < baWindowSize:
            print("Not enough keyframes for Bundle Adjustment")
            continue
        # Optimization parameters list (6 × (N−1)) + (3 × M) parameters, N number of keyframes, M number of landmarks of those keyframes
        # Will also need which landmarks correspond to which keyframes
        poseOptimizationParams = []
        landmarkOptimizationParams = []
        landmarksIdsUsed = []
        frameToLandmarkIdToPixel = {i: {} for i in range(baWindowSize)}
        for i, keyframeFrameId in enumerate(state.keyframesIds[-baWindowSize:]):
            frame = dataset.frames[keyframeFrameId]
            
            if i != 0:
                rotationMatrix = frame.pose.R_cw
                rotationVector = Rotation.from_matrix(rotationMatrix).as_rotvec()
                translation = frame.pose.t_cw
                poseOptimizationParams.append(rotationVector[0])
                poseOptimizationParams.append(rotationVector[1])
                poseOptimizationParams.append(rotationVector[2])
                poseOptimizationParams.append(translation[0])
                poseOptimizationParams.append(translation[1])
                poseOptimizationParams.append(translation[2])
            
            for landmarkId, keypointIdx in frame.observations:
                frameToLandmarkIdToPixel[i][landmarkId] = frame.keypoints[keypointIdx].pt
                if landmarkId in landmarksIdsUsed:
                    continue
                landmark = state.map_points[landmarkId]
                landmarkOptimizationParams.append(landmark.position[0])
                landmarkOptimizationParams.append(landmark.position[1])
                landmarkOptimizationParams.append(landmark.position[2])
                landmarksIdsUsed.append(landmarkId)

        res = least_squares(fun=projectionResiduals, x0=poseOptimizationParams+landmarkOptimizationParams, args=(frameToLandmarkIdToPixel, landmarksIdsUsed, dataset.frames[state.keyframesIds[-baWindowSize]].pose, camera.K), method='trf')
        
        # Get optimization results
        for i, keyframeFrameId in enumerate(state.keyframesIds[-baWindowSize:]):
            frame = dataset.frames[keyframeFrameId]
            
            rotationMatrix = np.zeros((3,3))
            translation = np.zeros(3)
            if i == 0:
                continue
            else:
                rotationMatrix = Rotation.from_rotvec([res.x[(i-1)*6 + 0], res.x[(i-1)*6 + 1], res.x[(i-1)*6 + 2]]).as_matrix()
                translation = np.array([res.x[(i-1)*6 + 3], res.x[(i-1)*6 + 4], res.x[(i-1)*6 + 5]]).reshape(3,)
            frame.pose.R_cw = rotationMatrix
            frame.pose.t_cw = translation
            state.trajectory[frame.trajectoryIndex] = frame.pose
            
            for j in frameToLandmarkIdToPixel.keys():
                for lid, pixel in frameToLandmarkIdToPixel[j].items():
                    landmark_W = [res.x[6*(baWindowSize-1) + landmarksIdsUsed.index(lid)*3 + 0], res.x[6*(baWindowSize-1) + landmarksIdsUsed.index(lid)*3 + 1], res.x[6*(baWindowSize-1) + landmarksIdsUsed.index(lid)*3 + 2]]
                    state.map_points[landmarkId].position = np.array(landmark_W)
                    
                    
    plotTrajectory(state, camera, dataset)    
    plotPointCloud(state)
        
if __name__ == "__main__":
    main()