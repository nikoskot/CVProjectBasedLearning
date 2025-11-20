import cv2 as cv
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from Calibration import stereoCameraCalibration as stereoCalib
from Calibration import cameraCalibration as monoCalib
import os
import pathlib
import plotly.graph_objs as pgo
import plotly.offline as pyo
import tqdm
import time
import configargparse
import yaml
from datetime import datetime

def getParser():
    parser = configargparse.ArgParser(default_config_files=["Depth\depthEstimationConfig.yaml"])
    parser.add("--configFile", is_config_file=True, help='config file path')
    # parser = argparse.ArgumentParser(description="Camera Calibration")
    parser.add("--imagesFolder", type=lambda p: pathlib.Path(p).resolve(), default="Depth\depthEstimationImages")
    parser.add("--liveCapture", action="store_true")
    parser.add("--imagesGroup", type=str, choices=["all", "left", "right"], default="left")
    parser.add("--patternRowCorners", type=int, default=9)
    parser.add("--patternColumnCorners", type=int, default=6)
    parser.add("--dontRefineCorners", action="store_true")
    parser.add("--resultsSavePath", type=lambda p: pathlib.Path(p).resolve(), default="Depth\depthEstimationResults")
    parser.add("--calibrationParamsFile", type=lambda p: pathlib.Path(p).resolve(), default="Calibration\calibrationResults")
    return parser

def saveArgsToYaml(args, filename):
    # Convert Namespace to dict
    args_dict = vars(args)
    # Dump to YAML file
    with open(filename, 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False)
        
def draw_epipolar_lines(left, right, step=50):
    left_vis = left.copy()
    right_vis = right.copy()
    for y in range(0, left.shape[0], step):
        cv.line(left_vis,  (0, y), (left.shape[1], y), (0, 255, 0), 1)
        cv.line(right_vis, (0, y), (right.shape[1], y), (0, 255, 0), 1)
    return np.hstack((left_vis, right_vis))

def calcluateRectificationMappings(lK, lD, rK, rD, R, T, imageShape):
    h = imageShape[0]
    w = imageShape[1]

    lR, rR, lP, rP, Q, lRoi, rRoi = cv.stereoRectify(
        lK, lD, rK, rD, (w, h), R, T, alpha=-1
    )
    
    map1x, map1y = cv.initUndistortRectifyMap(lK, lD, lR, lP, (w, h), cv.CV_32FC1)
    map2x, map2y = cv.initUndistortRectifyMap(rK, rD, rR, rP, (w, h), cv.CV_32FC1)

    return map1x, map1y, map2x, map2y

def rectifyStereoImages(leftImages, rightImages, map1x, map1y, map2x, map2y):

    rectifiedLeft = []
    for i, img in enumerate(leftImages):
        rectified = cv.remap(img, map1x, map1y, cv.INTER_LINEAR)
        rectifiedLeft.append(rectified)
        # cv.imshow(f"Original vs Rectified left images {i}", np.hstack((img, rectified)))
        # cv.waitKey(500)
        # cv.destroyAllWindows()

    rectifiedRight = []
    for i, img in enumerate(rightImages):
        rectified = cv.remap(img, map2x, map2y, cv.INTER_LINEAR)
        rectifiedRight.append(rectified)
        # cv.imshow(f"Original vs Rectified right images {i}", np.hstack((img, rectified)))
        # cv.waitKey(500)
        # cv.destroyAllWindows()

    return rectifiedLeft, rectifiedRight

def calculateDepthMap(calibrationParams, leftImages, rightImages):
    
    map1x, map1y, map2x, map2y = calcluateRectificationMappings(calibrationParams['leftCameraMatrix'], calibrationParams['leftDistortion'], calibrationParams['rightCameraMatrix'], calibrationParams['rightDistortion'], calibrationParams['R'], calibrationParams['T'], leftImages[0].shape)

    rectifiedLeft, rectifiedRight = rectifyStereoImages(leftImages, rightImages, map1x, map1y, map2x, map2y)

    # ------------------
    left_matcher = cv.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 10,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=7,
        P1=8 * 3 * 7 ** 2,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * 7 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=150,
        speckleRange=1,
        preFilterCap=63,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 8000
    sigma = 1.5

    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    for (left, right) in zip(rectifiedLeft, rectifiedRight):
        vis = draw_epipolar_lines(left, right)
        cv.imshow('Rectified Pair and disparity Map', vis)
        cv.waitKey(0)

        imgL = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
        imgR = cv.cvtColor(right, cv.COLOR_BGR2GRAY)
        displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
        dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
        filteredImg = cv.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv.NORM_MINMAX)
        filteredImg = np.uint8(filteredImg)
        cv.imshow('Filtered disparity map', filteredImg)
        cv.waitKey(0)
        cv.destroyAllWindows()
    # ------------------

    # stereo = cv.StereoBM.create(numDisparities=16*10, blockSize=15)
    # stereo = cv.StereoSGBM.create(
    #     minDisparity=0,
    #     numDisparities=16*10,
    #     blockSize=7,
    #     P1=8*3*9**2,
    #     P2=32*3*9**2,
    #     mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    # )
    # for (left, right) in zip(rectifiedLeft, rectifiedRight):
    #     vis = draw_epipolar_lines(left, right)
    #     cv.imshow('Rectified Pair and disparity Map', vis)
    #     cv.waitKey(0)
    #     disparity = stereo.compute(cv.cvtColor(left, cv.COLOR_BGR2GRAY), cv.cvtColor(right , cv.COLOR_BGR2GRAY))
    #     disparity = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX)
    #     disparity = np.uint8(disparity)
    #     cv.imshow('Disparity map', disparity)
    #     cv.waitKey(0)

def loadCalibrationParams(file):
    config = {}
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()

            if key in ('cam0', 'cam1'):
                # Parse matrix inside brackets [ ... ]
                value = value.strip('[]')
                # Split rows by semicolon, columns by space
                rows = value.split(';')
                matrix = []
                for row in rows:
                    # Split by whitespace, convert to float
                    row_vals = list(map(float, row.split()))
                    matrix.append(row_vals)
                if key == 'cam0':
                    config['leftCameraMatrix'] = np.array(matrix)
                else:
                    config['rightCameraMatrix'] = np.array(matrix)
            else:
                # Try to convert to float or int automatically
                try:
                    if '.' in value:
                        config[key] = float(value)
                    else:
                        config[key] = int(value)
                except ValueError:
                    config[key] = value
    config['R'] = np.eye(3)
    config['T'] = np.array([config['baseline'] / 1000.0, 0, 0])
    config['leftDistortion'] = np.zeros(4)
    config['rightDistortion'] = np.zeros(4)
    return config


def main():
    parser = getParser()
    args = parser.parse_args()

    # Create necessary folders/paths
    print("---Creating path for depth estimation results.---")
    args.resultsSavePath = pathlib.Path.joinpath(args.resultsSavePath, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(args.resultsSavePath, exist_ok=True)
    # Save arguments use to file
    saveArgsToYaml(args, pathlib.Path.joinpath(args.resultsSavePath, "config.yaml"))
    
    print(f"---Starting depth estimation with: \n {vars(args)}---")
    
    leftImages, rightImages = [], []
    if args.liveCapture:
        print("---Capturing images from camera.---")
        # images = captureImagesFromStereoCameras()
    else:
        print(f"---Loading images from folder {args.imagesFolder}.---")
        leftImages, rightImages = monoCalib.loadImages(group="all", folderName=args.imagesFolder)
    if len(leftImages) == 0 or len(rightImages) == 0:
        print("No images to use. Quitting.")
        return
    
    print(f"---Load calibration parameters from {args.calibrationParamsFile}.---")
    calibrationParams = loadCalibrationParams(args.calibrationParamsFile)
    print(calibrationParams)
    # monoCalib.showImagesInGrid(leftImages)
    # monoCalib.showImagesInGrid(rightImages)

    # lK, lD, rK, rD, R, T, E, F = stereoCalib.stereoCameraCalibration(leftImages, rightImages, 9, 6, True)

    depthMap = calculateDepthMap(calibrationParams, leftImages, rightImages)
    
    # print(f"---Saving depth map to folder {args.resultsSavePath}---")
    
    
if __name__ == '__main__':
    main()
    