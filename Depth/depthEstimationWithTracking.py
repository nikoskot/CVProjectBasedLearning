import cv2 as cv
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from Calibration import stereoCameraCalibration as stereoCalib
import os
import pathlib
import tqdm
import configargparse
import yaml
from datetime import datetime

def getParser():
    parser = configargparse.ArgParser(default_config_files=["Depth\depthEstimationWithTrackingConfig.yaml"])
    parser.add("--configFile", is_config_file=True, help='config file path')
    # parser = argparse.ArgumentParser(description="Camera Calibration")
    parser.add("--resultsSavePath", type=lambda p: pathlib.Path(p).resolve(), default="Depth\depthEstimationResults")
    parser.add("--calibrationParamsFile", type=lambda p: pathlib.Path(p).resolve(), default="Calibration\stereoCalibrationResults")
    return parser

def saveArgsToYaml(args, filename):
    # Convert Namespace to dict
    args_dict = vars(args)
    # Dump to YAML file
    with open(filename, 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False)

def calcluateRectificationMappings(lK, lD, rK, rD, R, T, imageShape):
    h = imageShape[0]
    w = imageShape[1]

    lR, rR, lP, rP, Q, lRoi, rRoi = cv.stereoRectify(
        lK, lD, rK, rD, (w, h), R, T, alpha=-1
    )
    
    map1x, map1y = cv.initUndistortRectifyMap(lK, lD, lR, lP, (w, h), cv.CV_32FC1)
    map2x, map2y = cv.initUndistortRectifyMap(rK, rD, rR, rP, (w, h), cv.CV_32FC1)

    return map1x, map1y, map2x, map2y

def rectifyStereoFrames(leftFrame, rightFrame, map1x, map1y, map2x, map2y):

    rectifiedLeftFrame = cv.remap(leftFrame, map1x, map1y, cv.INTER_LINEAR)
    rectifiedRightFrame = cv.remap(rightFrame, map2x, map2y, cv.INTER_LINEAR)

    return rectifiedLeftFrame, rectifiedRightFrame

def drawEpipolarLines(left, right, step=50):
    leftCopy = cv.resize(left.copy(), (640, 360))
    rightCopy = cv.resize(right.copy(), (640, 360))
    for y in range(0, left.shape[0], step):
        cv.line(leftCopy,  (0, y), (left.shape[1], y), (0, 255, 0), 1)
        cv.line(rightCopy, (0, y), (right.shape[1], y), (0, 255, 0), 1)
    return np.hstack((rightCopy, leftCopy))

def depthWithTracking(calibrationParams):
    print(f"---Initializing cameras.---")
    api = cv.CAP_MSMF
    cameraWidth = 1920
    cameraHeight = 1080
    cameraFps = 60
    
    cap0 = cv.VideoCapture(1)
    if not cap0.isOpened():
        print("Cannot open camera 0.")
        return
    else: 
        print("Camera 0 opened.")
    cap0.set(cv.CAP_PROP_FRAME_WIDTH, cameraWidth)   # width in pixels
    cap0.set(cv.CAP_PROP_FRAME_HEIGHT, cameraHeight)   # height in pixels
    cap0.set(cv.CAP_PROP_FPS, cameraFps)  # frames per second
    
    cap1 = cv.VideoCapture(0)
    if not cap1.isOpened():
        print("Cannot open camera 1.")
        return
    else: 
        print("Camera 1 opened.")
    cap1.set(cv.CAP_PROP_FRAME_WIDTH, cameraWidth)   # width in pixels
    cap1.set(cv.CAP_PROP_FRAME_HEIGHT, cameraHeight)   # height in pixels
    cap1.set(cv.CAP_PROP_FPS, cameraFps)

    info0 = "Camera 0 (Left). 'q' to stop"
    info1 = "Camera 1 (Right). 'q' to stop"
    
    trackerInitialized = False
    
    while True:
        
        # Capture and show camera frames
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not ret0:
            print("Can't receive frame from camera 0 (stream end?). Exiting ...")
            break
        if not ret1:
            print("Can't receive frame from camera 1 (stream end?). Exiting ...")
            break

        frame0Copy = cv.resize(frame0.copy(), (640, 360))
        frame1Copy = cv.resize(frame1.copy(), (640, 360))
        # cv.imshow(info0, frame0Copy)
        # cv.imshow(info1, frame1Copy)
        
        # Rectify images
        map1x, map1y, map2x, map2y = calcluateRectificationMappings(calibrationParams['leftCameraMatrix'], calibrationParams['leftDistortionCoeffs'], calibrationParams['rightCameraMatrix'], calibrationParams['rightDistortionCoeffs'], calibrationParams['R'], calibrationParams['T'], frame0.shape)
        rectifiedLeft, rectifiedRight = rectifyStereoFrames(frame0, frame1, map1x, map1y, map2x, map2y)
        
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit on 'q'
            break
        if key == ord('t'):  # 't' to initialize tracker with object
            # Select object
            bbox = cv.selectROI("Tracker initialization", rectifiedLeft)
            cv.destroyWindow("Tracker initialization")

            tracker = cv.TrackerCSRT_create()
            tracker.init(rectifiedLeft, bbox)
            
            disparityCalculator = cv.StereoSGBM_create(
                                    minDisparity=0,
                                    numDisparities=320,
                                    blockSize=5,
                                    P1=8 * 3 * 5**2,
                                    P2=32 * 3 * 5**2,
                                    disp12MaxDiff=1,
                                    uniquenessRatio=10,
                                    speckleWindowSize=100,
                                    speckleRange=32
)
            
            trackerInitialized = True
        
        if trackerInitialized:
            success, bbox = tracker.update(rectifiedLeft)

            if success:
                x, y, w, h = map(int, bbox)
                cv.rectangle(rectifiedLeft, (x,y), (x+w,y+h), (0,255,0), 2)
                
                objCenterLeftX, objCenterLeftY = (x + w//2, y + h//2)
                
                lY = int(objCenterLeftY)
                lX = int(objCenterLeftX)
                
                # ------------Manual search----------------
                patchHalfSize = 100
                template = rectifiedLeft[lY-patchHalfSize:lY+patchHalfSize, lX-patchHalfSize:lX+patchHalfSize]

                bestXRight = None
                bestScore = float('inf')

                maxDisparity = 1000
                for rX in range(max(lX - maxDisparity, patchHalfSize), lX):
                    patch = rectifiedRight[lY-patchHalfSize:lY+patchHalfSize, rX-patchHalfSize:rX+patchHalfSize]
                    score = np.sum((template - patch)**2)
                    if score < bestScore:
                        bestScore = score
                        bestXRight = rX
                rX = bestXRight
                
                
                # --------------Search with disparity calculator-------------
                # disparity = disparityCalculator.compute(rectifiedLeft, rectifiedRight)
                # disparity = disparity / 16.0
                # d = disparity[lY, lX]
                # print(d)
                # rX = int(lX - d)
                
                cv.drawMarker(rectifiedLeft, (lX, lY), (0,0,255), markerSize=30, thickness=4)
                cv.drawMarker(rectifiedRight, (rX, lY), (0,0,255), markerSize=30, thickness=4)
                # disp_vis = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX)
                # disp_vis = disp_vis.astype(np.uint8)
                # cv.imshow("Disparity", cv.resize(disp_vis, (640, 380)))
                
            else:
                # cv.putText(frame, "Tracking failure", (20,40),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                pass
        
        cv.imshow(info0, frame0Copy)
        cv.imshow(info1, frame1Copy)
        
        # Show rectified frames side by side with epipolar lines
        cv.imshow("Rectified, scaled down, with epipolar lines", drawEpipolarLines(rectifiedLeft, rectifiedRight))
            
        
    cap0.release()
    cap1.release()
    cv.destroyAllWindows()
        
def main():
    parser = getParser()
    args = parser.parse_args()

    # Create necessary folders/paths
    print("---Creating path for depth estimation results.---")
    args.resultsSavePath = pathlib.Path.joinpath(args.resultsSavePath, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(args.resultsSavePath, exist_ok=True)
    # Save arguments use to file
    saveArgsToYaml(args, pathlib.Path.joinpath(args.resultsSavePath, "config.yaml"))
    
    print("---Starting depth estimation with tracking with:")
    print('\n'.join(f"{k}: {v}" for k, v in vars(args).items()))
    
    print(f"---Load calibration parameters from {args.calibrationParamsFile}.---")
    calibrationParams = stereoCalib.loadCalibrationParams(args.calibrationParamsFile)
    print(f"Loaded calibration parameters: \n {calibrationParams}")

    depthWithTracking(calibrationParams)
        
    
if __name__ == '__main__':
    main()