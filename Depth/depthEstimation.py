import cv2 as cv
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from Calibration import cameraCalibration as monoCalib
import os
import pathlib
import tqdm
import configargparse
import yaml
from datetime import datetime

def getParser():
    parser = configargparse.ArgParser(default_config_files=["Depth\depthEstimationConfig.yaml"])
    parser.add("--configFile", is_config_file=True, help='config file path')
    # parser = argparse.ArgumentParser(description="Camera Calibration")
    parser.add("--imagesFolder", type=lambda p: pathlib.Path(p).resolve(), default="Depth\depthEstimationImages")
    parser.add("--liveCapture", action="store_true")
    parser.add("--resultsSavePath", type=lambda p: pathlib.Path(p).resolve(), default="Depth\depthEstimationResults")
    parser.add("--calibrationParamsFile", type=lambda p: pathlib.Path(p).resolve(), default="Calibration\calibrationResults")
    parser.add("--rectifyImages", action="store_true", help="Whether to rectify images before depth estimation")
    parser.add("--minDisparity", type=int, default=0, help="Minimum possible disparity value")
    parser.add("--maxDisparity", type=int, default=64, help="Maximum possible disparity value. Must be divisible by 16.")
    parser.add("--blockSize", type=int, default=9, help="Matched block size. It must be an odd number >=1 ")
    parser.add("--preFilterCap", type=int, default=31, help="Truncation value for the prefiltered image pixels")
    parser.add("--uniquenessRatio", type=int, default=10, help="Margin in percentage by which the best (minimum) computed cost function value should 'win' the second best value to consider the found match correct")
    parser.add("--speckleWindowSize", type=int, default=100, help="Maximum size of smooth disparity regions to consider their noise speckles and invalidate")
    parser.add("--speckleRange", type=int, default=2, help="Maximum disparity variation within each connected component")
    parser.add("--disp12MaxDiff", type=int, default=1, help="Maximum allowed difference in the left-right disparity check")
    parser.add("--wlsLambda", type=float, default=8000.0, help="Amount of regularization during filtering")
    parser.add("--wlsSigma", type=float, default=1.5, help="Standard deviation of the color filter that is used during filtering")
    return parser

def saveArgsToYaml(args, filename):
    # Convert Namespace to dict
    args_dict = vars(args)
    # Dump to YAML file
    with open(filename, 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False)
        
def drawEpipolarLines(left, right, step=50):
    leftCopy = left.copy()
    rightCopy = right.copy()
    for y in range(0, left.shape[0], step):
        cv.line(leftCopy,  (0, y), (left.shape[1], y), (0, 255, 0), 1)
        cv.line(rightCopy, (0, y), (right.shape[1], y), (0, 255, 0), 1)
    return np.hstack((leftCopy, rightCopy))

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

def calculateDepthMap(args, calibrationParams, leftImages, rightImages):
    
    rectifiedLeft, rectifiedRight = leftImages, rightImages
    if args.rectifyImages:
        map1x, map1y, map2x, map2y = calcluateRectificationMappings(calibrationParams['leftCameraMatrix'], calibrationParams['leftDistortion'], calibrationParams['rightCameraMatrix'], calibrationParams['rightDistortion'], calibrationParams['R'], calibrationParams['T'], leftImages[0].shape)
        rectifiedLeft, rectifiedRight = rectifyStereoImages(leftImages, rightImages, map1x, map1y, map2x, map2y)

    stereo = cv.StereoSGBM_create(
        minDisparity = args.minDisparity,
        numDisparities = args.maxDisparity - args.minDisparity,
        blockSize= args.blockSize,
        preFilterCap=args.preFilterCap,
        uniquenessRatio = args.uniquenessRatio,
        speckleWindowSize = args.speckleWindowSize,
        speckleRange = args.speckleRange,
        disp12MaxDiff = args.disp12MaxDiff,
        P1 = 8 * 1 * args.blockSize**2,
        P2 = 32 * 1 * args.blockSize**2,
        mode=cv.STEREO_SGBM_MODE_SGBM
    )

    leftMatcher = stereo
    rightMatcher = cv.ximgproc.createRightMatcher(leftMatcher)

    disparity_filter = cv.ximgproc.createDisparityWLSFilter(leftMatcher)
    disparity_filter.setLambda(args.wlsLambda)
    disparity_filter.setSigmaColor(args.wlsSigma)

    for i, (left, right) in tqdm.tqdm(enumerate(zip(rectifiedLeft, rectifiedRight)), total=len(rectifiedLeft)):
        
        epipolarLines = drawEpipolarLines(left, right)
        filePath = pathlib.Path.joinpath(args.resultsSavePath, 'epipolarLinesImages', f'{i}.png')
        print(f"\nSaving epipolarlines visualization to {filePath}")
        cv.imwrite(filePath, epipolarLines)

        leftGray, rightGray = left, right
        if len(left.shape) == 3:
            leftGray = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
            rightGray = cv.cvtColor(right, cv.COLOR_BGR2GRAY)

        disparityLeft = leftMatcher.compute(leftGray, rightGray)
        disparityRight = rightMatcher.compute(rightGray, leftGray)

        disparityLeft = np.int16(disparityLeft)
        disparityRight = np.int16(disparityRight)
        
        disparityFiltered = disparity_filter.filter(disparityLeft, leftGray, None, disparityRight)

        disparity = cv.normalize(disparityFiltered, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        filePath = pathlib.Path.joinpath(args.resultsSavePath, 'depthMaps', f'{i}.png')
        print(f"Saving depth map to {filePath}")
        cv.imwrite(filePath, disparity)

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
    os.makedirs(pathlib.Path.joinpath(args.resultsSavePath, 'epipolarLinesImages'), exist_ok=True)
    os.makedirs(pathlib.Path.joinpath(args.resultsSavePath, 'depthMaps'), exist_ok=True)
    # Save arguments use to file
    saveArgsToYaml(args, pathlib.Path.joinpath(args.resultsSavePath, "config.yaml"))
    
    print("---Starting depth estimation with:")
    print('\n'.join(f"{k}: {v}" for k, v in vars(args).items()))

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

    print("---Calculating depth maps.---")
    calculateDepthMap(args, calibrationParams, leftImages, rightImages)
    
if __name__ == '__main__':
    main()
    