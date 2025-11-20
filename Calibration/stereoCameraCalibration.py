import cv2 as cv
import os
import numpy as np
from scipy.spatial.transform import Rotation
import rerun as rr

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from Utils import utils
from Calibration.cameraCalibration import monocularCameraCalibration, opencvSingleCameraCalibration, loadImages, showImagesInGrid



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

    print(f"Stereo calibration RMSE: \n{ret}")
    print(f"Left Camera Matrix after Stereo Calibration: \n{lK}")
    print(f"Left Camera Distortion coefficients after Stereo Calibration: \n{lD}")
    print(f"Right Camera Matrix after Stereo Calibration: \n{rK}")
    print(f"Right Camera Distortion coefficients after Stereo Calibration: \n{rD}")
    print(f"Rotation between cameras: \n{R}")
    print(f"Translation between cameras: \n{T}")
    print(f"Essential Matrix: \n{E}")
    print(f"Fundamental Matrix: \n{F}")
    
    return lK, lD, rK, rD, R, T, E, F

def manualStereoCameraCalibration(leftImages, rightImages, nCornersPerRow=9, nCornersPerColumn=6, refineCorners=True):

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

    Ris = []
    Tis = []
    for i in range(len(leftImages)):
        Ris.append(Rotation.from_rotvec(rRs[i].squeeze()).as_matrix() @ Rotation.from_rotvec(lRs[i].squeeze()).as_matrix().T)
    Ris = np.stack(Ris)
    rots = Rotation.from_matrix(Ris).as_rotvec()
    R = Rotation.from_rotvec(rots.mean(axis=0)).as_matrix()
    print(f"Rotation between cameras: \n{R}")

    # A = np.eye(3) * len(lTs)
    b = sum(t2 - R @ t1 for t1, t2 in zip(lTs, rTs))
    T = b / len(lTs)
    print(f"Translation between cameras: \n{T}")

    # -------------------- Optimize --------------------------------
    # from scipy.optimize import least_squares

    # def optimize_stereo_RT(lRs, lTs, rRs, rTs, R_init=None, T_init=None):
    #     """
    #     Optimize relative rotation R and translation T such that:
    #     Rr_i ~ R @ Rl_i
    #     tr_i ~ R @ tl_i + T
    #     Inputs can be rotation vectors (3,) or rotation matrices (3,3).
    #     Returns R_opt (3x3), T_opt (3,), and the OptimizeResult.
    #     """
    #     # convert to consistent numpy shapes
    #     lRm = [utils._to_rotmat(r) for r in lRs]
    #     rRm = [utils._to_rotmat(r) for r in rRs]
    #     ltv = [utils._to_vec(t) for t in lTs]
    #     rtv = [utils._to_vec(t) for t in rTs]

    #     # initial guess
    #     if R_init is None:
    #         Rrels = [rR @ lR.T for rR, lR in zip(rRm, lRm)]
    #         rots = Rotation.from_matrix(np.stack(Rrels))
    #         r_init = rots.as_rotvec().mean(axis=0)
    #     else:
    #         R_init = np.asarray(R_init)
    #         r_init = Rotation.from_matrix(R_init).as_rotvec() if R_init.shape == (3,3) else R_init.reshape(3,)

    #     if T_init is None:
    #         # use average: tr - Rrel*tl  (Rrel ~ rR @ lR.T)
    #         Rrels = [rR @ lR.T for rR, lR in zip(rRm, lRm)]
    #         T_init = np.mean([r - Rrel @ l for r, l, Rrel in zip(rtv, ltv, Rrels)], axis=0)
    #     x0 = np.hstack([r_init, T_init])

    #     def residuals(x):
    #         R = Rotation.from_rotvec(x[:3]).as_matrix()
    #         T = x[3:6]
    #         res = []
    #         for lR, lT, rR, rT in zip(lRm, ltv, rRm, rtv):
    #             R_pred = R @ lR
    #             # rotation error as small rotation vector: log( rR.T * R_pred )
    #             R_err_mat = rR.T @ R_pred
    #             r_err = Rotation.from_matrix(R_err_mat).as_rotvec()
    #             res.extend(r_err.tolist())
    #             t_err = (rT - (R @ lT + T)).tolist()
    #             res.extend(t_err)
    #         return np.array(res)

    #     result = least_squares(residuals, x0, method='lm')
    #     R_opt = Rotation.from_rotvec(result.x[:3]).as_matrix()
    #     T_opt = result.x[3:6]
    #     print(f"Optimized Rotation between cameras: \n{R_opt}")
    #     print(f"Optimized Translation between cameras: \n{T_opt}")

    # optimize_stereo_RT(lRs, lTs, rRs, rTs)


    return R, T

def visualizeSetup(R, T, K1, K2):
    # rr.init("stereo_calibration", spawn=True)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # Left camera (world origin)
    rr.log("world/left_cam", rr.Transform3D(mat3x3=R, translation=T.flatten()), static=True)
    # rr.log("left_cam/frame", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/left_cam/axes", rr.Arrows3D(origins=np.zeros((3, 3)), vectors=np.eye(3), colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]), static=True)
    rr.log("world/left_cam/image", rr.Pinhole(image_from_camera=K1, resolution=[640, 480]), static=True)
    # Right camera
    rr.log("world/right_cam", rr.Transform3D(mat3x3=np.eye(3), translation=[0,0,0]), static=True)
    rr.log("world/right_cam/axes", rr.Arrows3D(origins=np.zeros((3, 3)), vectors=np.eye(3), colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]), static=True)
    rr.log("world/right_cam/image", rr.Pinhole(image_from_camera=K2, resolution=[640, 480]), static=True)

def stereoCameraCalibration(leftImages, rightImages, nCornersPerRow=9, nCornersPerColumn=6, refineCorners=True):
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

    print(f"Stereo calibration RMSE: \n{ret}")
    print(f"Left Camera Matrix after Stereo Calibration: \n{lK}")
    print(f"Left Camera Distortion coefficients after Stereo Calibration: \n{lD}")
    print(f"Right Camera Matrix after Stereo Calibration: \n{rK}")
    print(f"Right Camera Distortion coefficients after Stereo Calibration: \n{rD}")
    print(f"Rotation between cameras: \n{R}")
    print(f"Translation between cameras: \n{T}")
    print(f"Essential Matrix: \n{E}")
    print(f"Fundamental Matrix: \n{F}")
    
    return lK, lD, rK, rD, R, T, E, F

if __name__ == '__main__':

    leftImages, rightImages = loadImages("all")
    showImagesInGrid(leftImages)
    showImagesInGrid(rightImages)

    # stereoCameraCalibration(leftImages, rightImages, 9, 6, True)
    print("------------ OpenCV -------------")
    leftK, leftDist, rightK, rightDist, R, T, E, F = openCVStereoCameraCalibration(leftImages, rightImages, 9, 6, True)

    # rr.init("stereo_calibration", spawn=True)
    # visualizeSetup(R, T, leftK, rightK)

    print("------------ Manual -------------")
    manualStereoCameraCalibration(leftImages, rightImages, 9, 6, True)
    