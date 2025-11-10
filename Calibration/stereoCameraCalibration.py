import cv2 as cv
import os
import numpy as np
from cameraCalibration import singleCameraCalibration, opencvSingleCameraCalibration, loadCalibrationImages, showImagesInGrid

def stereoCameraCalibration(leftImages, rightImages, nCornersPerRow=9, nCornersPerColumn=6, refineCorners=True):
    
    lK, lRs, lTs, lDistortionCoeffs = singleCameraCalibration(leftImages, nCornersPerRow, nCornersPerColumn, refineCorners)
    rK, rRs, rTs, rDistortionCoeffs = singleCameraCalibration(rightImages, nCornersPerRow, nCornersPerColumn, refineCorners)

    print(f"Left Camera Matrix: \n{lK}")
    print(f"Left Camera Distortion coefficients: \n{lDistortionCoeffs}")
    
    print(f"Right Camera Matrix: \n{rK}")
    print(f"Right Camera Distortion coefficients: \n{rDistortionCoeffs}")


def openCVStereoCameraCalibration(leftImages, rightImages, nCornersPerRow=9, nCornersPerColumn=6, refineCorners=True):

    worldCoordsSingle = np.zeros((nCornersPerRow*nCornersPerColumn, 3), np.float32)
    worldCoordsSingle[:, :2] = np.mgrid[0:nCornersPerRow, 0:nCornersPerColumn].T.reshape(-1, 2)
    leftImageCoords = [] 
    worldCoords = []
    h, w, _ = leftImages[0].shape
    
    for i, img in enumerate(leftImages):
        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(img, (nCornersPerRow, nCornersPerColumn))
        if not ret:
            print(f"No chessboard corners found in image {i}")
            cv.imshow(f"No corners {i}", img)
            cv.waitKey(1000)
            cv.destroyAllWindows()
            continue
        
        if refineCorners:
            corners = cv.cornerSubPix(cv.cvtColor(img, cv.COLOR_BGR2GRAY), corners, (11,11), (-1,-1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            # corners = cv.cornerSubPix(img, corners, (11,11), (-1,-1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        img_copy = img.copy()
        cv.drawChessboardCorners(img_copy, (nCornersPerRow, nCornersPerColumn), corners, ret)
        cv.imshow(f"Annotated corners {i}", img_copy)
        cv.waitKey(500)
        cv.destroyAllWindows()

        leftImageCoords.append(corners.reshape(-1, 2))
        worldCoords.append(worldCoordsSingle)

    rightImageCoords = [] 

    for i, img in enumerate(rightImages):
        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(img, (nCornersPerRow, nCornersPerColumn))
        if not ret:
            print(f"No chessboard corners found in image {i}")
            cv.imshow(f"No corners {i}", img)
            cv.waitKey(1000)
            cv.destroyAllWindows()
            continue
        
        if refineCorners:
            corners = cv.cornerSubPix(cv.cvtColor(img, cv.COLOR_BGR2GRAY), corners, (11,11), (-1,-1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            # corners = cv.cornerSubPix(img, corners, (11,11), (-1,-1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        img_copy = img.copy()   
        cv.drawChessboardCorners(img_copy, (nCornersPerRow, nCornersPerColumn), corners, ret)
        cv.imshow(f"Annotated corners {i}", img_copy)
        cv.waitKey(500)
        cv.destroyAllWindows()

        rightImageCoords.append(corners.reshape(-1, 2))

    lK, lRs, lTs, lDistortionCoeffs = opencvSingleCameraCalibration(leftImages, worldCoords, leftImageCoords)
    rK, rRs, rTs, rDistortionCoeffs = opencvSingleCameraCalibration(rightImages, worldCoords, rightImageCoords)

    print(f"Left Camera Matrix: \n{lK}")
    print(f"Left Camera Distortion coefficients: \n{lDistortionCoeffs}")
    
    print(f"Right Camera Matrix: \n{rK}")
    print(f"Right Camera Distortion coefficients: \n{rDistortionCoeffs}")

    ret, lK, lD, rK, rD, R, T, E, F = cv.stereoCalibrate(
        worldCoords, leftImageCoords, rightImageCoords,
        lK, lDistortionCoeffs, rK, rDistortionCoeffs,
        (w, h),
        criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=cv.CALIB_FIX_INTRINSIC
    )

    print(f"Left Camera Matrix after Stereo Calibration: \n{lK}")
    print(f"Left Camera Distortion coefficients after Stereo Calibration: \n{lD}")
    print(f"Right Camera Matrix after Stereo Calibration: \n{rK}")
    print(f"Right Camera Distortion coefficients after Stereo Calibration: \n{rD}")
    print(f"Rotation between cameras: \n{R}")
    print(f"Translation between cameras: \n{T}")
    print(f"Essential Matrix: \n{E}")
    print(f"Fundamental Matrix: \n{F}")
    
    lR, rR, lP, rP, Q, lRoi, rRoi = cv.stereoRectify(
        lK, lD, rK, rD, (w, h), R, T, alpha=-1
    )
    
    map1x, map1y = cv.initUndistortRectifyMap(lK, lD, lR, lP, (w, h), cv.CV_32FC1)
    map2x, map2y = cv.initUndistortRectifyMap(rK, rD, rR, rP, (w, h), cv.CV_32FC1)

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
    
    # ------------------
    for dispFactor in [7, 10, 14]: # 10
        for ws in [7, 9, 11]: # any
            numDispFactor = dispFactor
            window_size = ws  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
            left_matcher = cv.StereoSGBM_create(
                minDisparity=0,
                numDisparities=16 * numDispFactor,  # max_disp has to be dividable by 16 f. E. HH 192, 256
                blockSize=window_size,
                P1=8 * 3 * window_size ** 2,
                # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
                P2=32 * 3 * window_size ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=15,
                speckleWindowSize=0,
                speckleRange=2,
                preFilterCap=63,
                mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
            )
            right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
            # FILTER Parameters
            lmbda = 80000
            sigma = 1.3

            wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
            wls_filter.setLambda(lmbda)

            imgL = cv.cvtColor(rectifiedLeft[0], cv.COLOR_BGR2GRAY)
            imgR = cv.cvtColor(rectifiedRight[0], cv.COLOR_BGR2GRAY)
            wls_filter.setSigmaColor(sigma)
            displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
            dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
            displ = np.int16(displ)
            dispr = np.int16(dispr)
            filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

            filteredImg = cv.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv.NORM_MINMAX)
            filteredImg = np.uint8(filteredImg)
            cv.imshow(f'Filtered disparity map {dispFactor} {ws}', filteredImg)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # ---------------------------------------------
    cv.imshow('LR', np.hstack((rectifiedLeft[0], rectifiedRight[0])))
    cv.imshow('L', rectifiedLeft[0])
    cv.imshow('R', rectifiedRight[0])
    cv.waitKey(0)
    cv.destroyAllWindows()

    for dispFactor in [7, 10, 14]:
        for ws in [5, 7, 9, 11]:
            for minD in [0, 4, 16]:
                minDisp = minD
                numDispFactor = dispFactor
                windowSize=ws
                stereo = cv.StereoSGBM.create(
                    minDisparity=minDisp,
                    numDisparities=16*numDispFactor-minDisp,
                    blockSize=windowSize,
                    P1=8*3*windowSize**2,
                    P2=32*3*windowSize**2,
                    disp12MaxDiff=1,
                    uniquenessRatio=15,
                    speckleWindowSize=0,
                    speckleRange=2,
                    preFilterCap=63,
                    mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
                )
                disparity = stereo.compute(cv.cvtColor(rectifiedLeft[0], cv.COLOR_BGR2GRAY), cv.cvtColor(rectifiedRight[0] , cv.COLOR_BGR2GRAY))
                disparity = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX)
                disparity = np.uint8(disparity)
                cv.imshow(f'Disparity map {dispFactor} {ws} {minD}', disparity)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # stereo = cv.StereoBM.create(numDisparities=16*10, blockSize=15)
    stereo = cv.StereoSGBM.create(
        minDisparity=0,
        numDisparities=16*12,
        blockSize=9,
        P1=8*3*9**2,
        P2=32*3*9**2,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )
    for (left, right) in zip(rectifiedLeft, rectifiedRight):
        vis = draw_epipolar_lines(left, right)
        cv.imshow('Rectified Pair and disparity Map', vis)
        cv.waitKey(0)
        disparity = stereo.compute(cv.cvtColor(left, cv.COLOR_BGR2GRAY), cv.cvtColor(right , cv.COLOR_BGR2GRAY))
        disparity = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX)
        disparity = np.uint8(disparity)
        cv.imshow('Disparity map', disparity)
        cv.waitKey(0)

    return lK, lD, rK, rD, R, T, E, F

def draw_epipolar_lines(left, right, step=50):
    left_vis = left.copy()
    right_vis = right.copy()
    for y in range(0, left.shape[0], step):
        cv.line(left_vis,  (0, y), (left.shape[1], y), (0, 255, 0), 1)
        cv.line(right_vis, (0, y), (right.shape[1], y), (0, 255, 0), 1)
    return np.hstack((left_vis, right_vis))

if __name__ == '__main__':

    leftImages, rightImages = loadCalibrationImages("all")
    showImagesInGrid(leftImages)
    showImagesInGrid(rightImages)

    # stereoCameraCalibration(leftImages, rightImages, 9, 6, True)
    print("-------------------------")
    openCVStereoCameraCalibration(leftImages, rightImages, 9, 6, True)
    