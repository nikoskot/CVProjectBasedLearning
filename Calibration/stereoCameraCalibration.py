import cv2 as cv
import os
import numpy as np
from scipy.spatial.transform import Rotation
import rerun as rr
import configargparse
import pathlib
from datetime import datetime
import yaml
import time
import sys
import tqdm
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from Utils import utils
from Calibration import cameraCalibration as monoCalib

def getParser():
    parser = configargparse.ArgParser(default_config_files=["Calibration\stereoCalibrationConfig.yaml"])
    parser.add("--configFile", is_config_file=True, help='config file path')
    # parser = argparse.ArgumentParser(description="Camera Calibration")
    parser.add("--imagesFolder", type=lambda p: pathlib.Path(p).resolve(), default="Calibration\calibrationImages")
    parser.add("--liveCapture", action="store_true")
    # parser.add("--imagesGroup", type=str, choices=["left", "right"], default="all")
    parser.add("--patternRowCorners", type=int, default=9)
    parser.add("--patternColumnCorners", type=int, default=6)
    parser.add("--dontRefineCorners", action="store_true")
    parser.add("--resultsSavePath", type=lambda p: pathlib.Path(p).resolve(), default="Calibration\stereoCalibrationResults")
    return parser

def saveArgsToYaml(args, filename):
    # Convert Namespace to dict
    args_dict = vars(args)
    # Dump to YAML file
    with open(filename, 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False)

def captureCalibrationImagesFromTwoCameras():
    
    cap0 = cv.VideoCapture(0)
    cap1 = cv.VideoCapture(1)
    if not (cap0.isOpened() and cap1.isOpened()):
        print("Cannot open cameras.")
        return

    cap0.set(cv.CAP_PROP_FRAME_WIDTH, 1920)   # width in pixels
    cap0.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)   # height in pixels
    cap1.set(cv.CAP_PROP_FRAME_WIDTH, 1920)   # width in pixels
    cap1.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)   # height in pixels

    save_interval = 3.0  # seconds between saves
    saving = False
    last_save_time = 0

    frame_count = 0  # count saved frames
    leftImages = []
    rightImages = []
    info0 = "Left Camera. 's' tp start saving. 'q' to stop"
    info1 = "Right Camera. 's' tp start saving. 'q' to stop"
    
    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not (ret0 and ret1):
            print("Can't receive frame from at least one camera (stream end?). Exiting ...")
            break

        current_time = time.time()
        frame0Copy = cv.flip(frame0.copy(), 1)
        frame1Copy = cv.flip(frame1.copy(), 1)

        # If saving mode started, check time and save frames every "save_interval" seconds
        if saving:
            elapsed = current_time - last_save_time

            # Calculate countdown (seconds remaining to next save)
            countdown = max(0, save_interval - elapsed)
            countdown_text = f"Next capture in: {countdown:.1f}s"

            # Put countdown text on frame
            cv.putText(frame0Copy, countdown_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv.LINE_AA)

            if elapsed >= save_interval:
                # Save frame
                leftImages.append(frame0)
                rightImages.append(frame1)
                frame_count += 1
                last_save_time = current_time

        else:
            # Show instruction
            cv.putText(frame0Copy, f"Press 's' to start saving every {save_interval} seconds", (10, 30),
                        cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)

        cv.imshow(info0, frame0Copy)
        cv.imshow(info1, frame0Copy)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit on 'q'
            break
        elif key == ord('s'):  # Start saving on 's'
            if not saving:
                cv.destroyAllWindows()
                saving = True
                last_save_time = current_time

    cap0.release()
    cap1.release()
    cv.destroyAllWindows()
    return leftImages, rightImages

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

def stereoCameraCalibration(leftImages, rightImages, nCornersPerRow=9, nCornersPerColumn=6, refineCorners=True, savePath=None):
    worldCoordsSingle = np.zeros((nCornersPerRow*nCornersPerColumn, 3), np.float32)
    worldCoordsSingle[:, :2] = np.mgrid[0:nCornersPerRow, 0:nCornersPerColumn].T.reshape(-1, 2)
    leftImageCoords = []
    rightImageCoords = [] 
    worldCoords = []
    h, w, _ = leftImages[0].shape
    
    print("Detecting pattern cornenrs.")
    for i in tqdm.tqdm(range(len(leftImages))):
        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(leftImages[i], (nCornersPerRow, nCornersPerColumn))
        if not ret:
            print(f"No chessboard corners found in left image {i}")
            cv.imshow(f"No corners {i}", leftImages[i])
            cv.waitKey(1000)
            cv.destroyAllWindows()
            continue
        
        if refineCorners:
            corners = cv.cornerSubPix(cv.cvtColor(leftImages[i], cv.COLOR_BGR2GRAY), corners, (11,11), (-1,-1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        if savePath:
            imgCopy = leftImages[i].copy()
            cv.drawChessboardCorners(imgCopy, (nCornersPerRow, nCornersPerColumn), corners, ret)
            cv.imwrite(pathlib.Path.joinpath(savePath, f"annotatedLeft{i}.png"), imgCopy)

        leftImageCoords.append(corners.reshape(-1, 2))
        worldCoords.append(worldCoordsSingle)

        ret, corners = cv.findChessboardCorners(rightImages[i], (nCornersPerRow, nCornersPerColumn))
        if not ret:
            print(f"No chessboard corners found in right image {i}")
            cv.imshow(f"No corners {i}", rightImages[i])
            cv.waitKey(1000)
            cv.destroyAllWindows()
            continue
        
        if refineCorners:
            corners = cv.cornerSubPix(cv.cvtColor(rightImages[i], cv.COLOR_BGR2GRAY), corners, (11,11), (-1,-1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        if savePath:
            imgCopy = rightImages[i].copy()
            cv.drawChessboardCorners(imgCopy, (nCornersPerRow, nCornersPerColumn), corners, ret)
            cv.imwrite(pathlib.Path.joinpath(savePath, f"annotatedRight{i}.png"), imgCopy)

        rightImageCoords.append(corners.reshape(-1, 2))

    print("Calibrating.")
    start = time.time()
    leftRmse, leftCameraMatrix, leftDistortionCoeffs, leftRotationVecs, leftTranslationVecs = monoCalib.opencvSingleCameraCalibration(leftImages, worldCoords, leftImageCoords)
    rightRmse, rightCameraMatrix, rightDistortionCoeffs, rightRotationVecs, rightTranslationVecs = monoCalib.opencvSingleCameraCalibration(leftImages, worldCoords, leftImageCoords)

    print(f"Left monocular calibration RMSE (pixels): \n{leftRmse}")
    print(f"Left camera matrix: \n{leftCameraMatrix}")
    print(f"Left camera distortion coefficients: \n{leftDistortionCoeffs}")
    
    print(f"Right monocular calibration RMSE (pixels): \n{rightRmse}")
    print(f"Right camera matrix: \n{rightCameraMatrix}")
    print(f"Right camera distortion coefficients: \n{rightDistortionCoeffs}")

    stereoRmse, _, _, _, _, R, T, E, F = cv.stereoCalibrate(
        worldCoords, 
        leftImageCoords, 
        rightImageCoords,
        leftCameraMatrix, 
        leftDistortionCoeffs, 
        rightCameraMatrix, 
        rightDistortionCoeffs,
        (w, h),
        flags=cv.CALIB_FIX_INTRINSIC
    )
    
    print(f"Stereo setup calibration took {time.time() - start}.")

    print(f"Stereo calibration RMSE: \n{stereoRmse}")
    print(f"Rotation between cameras: \n{R}")
    print(f"Translation between cameras: \n{T}")
    print(f"Essential Matrix: \n{E}")
    print(f"Fundamental Matrix: \n{F}")
    
    return stereoRmse, leftCameraMatrix, leftDistortionCoeffs, rightCameraMatrix, rightDistortionCoeffs, R, T, E, F

def saveCalibrationParams(folderPath, stereoRmse, leftCameraMatrix, leftDistortionCoeffs, rightCameraMatrix, rightDistortionCoeffs, R, T, E, F):
    try:
        fs = cv.FileStorage(pathlib.Path.joinpath(folderPath, "stereoCalib.json"), cv.FILE_STORAGE_WRITE)
        fs.write("leftCameraMatrix", leftCameraMatrix)
        fs.write("leftDistortionCoeffs", leftDistortionCoeffs)
        fs.write("rightCameraMatrix", rightCameraMatrix)
        fs.write("rightDistortionCoeffs", rightDistortionCoeffs)
        fs.write("rmse", stereoRmse)
        fs.write("R", R)
        fs.write("T", T)
        fs.write("E", E)
        fs.write("F", F)
        fs.release()
    except Exception as e:
        print(f"Could not save stereo calibration results in file {folderPath}\stereoCalib.json. \n Exception {e}")

def loadCalibrationParams(folderPath):
    calib_path = pathlib.Path(folderPath) / "stereoCalib.json"
    fs = cv.FileStorage(str(calib_path), cv.FILE_STORAGE_READ)

    if not fs.isOpened():
        raise FileNotFoundError(f"Could not open {calib_path}")

    data = {}

    root = fs.getFirstTopLevelNode()
    while not root.empty():
        key = root.name()
        if root.isReal():
            data[key] = root.real()
        elif root.isString():
            data[key] = root.string()
        else:
            data[key] = root.mat()
        root = root.next()

    fs.release()
    return data
        
def main():
    parser = getParser()
    args = parser.parse_args()

    # Create necessary folders/paths
    print("---Creating path for calibration results.---")
    args.resultsSavePath = pathlib.Path.joinpath(args.resultsSavePath, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(args.resultsSavePath, exist_ok=True)
    os.makedirs(pathlib.Path.joinpath(args.resultsSavePath, "annotatedImages"), exist_ok=True)
    # Save arguments use to file
    saveArgsToYaml(args, pathlib.Path.joinpath(args.resultsSavePath, "config.yaml"))

    print(f"---Starting stereo camera calibration with: \n {vars(args)}---")

    images = []
    if args.liveCapture:
        print("---Capturing calibration images from two cameras.---")
        leftImages, rightImages = captureCalibrationImagesFromTwoCameras()
    else:
        print(f"---Loading images from folder {args.imagesFolder}.---")
        leftImages, rightImages = monoCalib.loadImages(group="all", folderName=args.imagesFolder)
    if len(leftImages) == 0 or len(rightImages) == 0:
        print("At least one set of images is empty. Quitting.")
        return
    if len(leftImages) != len(rightImages):
        print("Left/Right set of images are not the same number. Quitting.")
        return
    
    # showImagesInGrid(leftImages)
    # showImagesInGrid(rightImages)

    stereoRmse, leftCameraMatrix, leftDistortionCoeffs, rightCameraMatrix, rightDistortionCoeffs, R, T, E, F = stereoCameraCalibration(
        leftImages=leftImages, 
        rightImages=rightImages, 
        nCornersPerRow=args.patternRowCorners, 
        nCornersPerColumn=args.patternColumnCorners, 
        refineCorners=(not args.dontRefineCorners), 
        savePath=pathlib.Path.joinpath(args.resultsSavePath, "annotatedImages")
        )

    print(f"---Saving calibration results to folder {args.resultsSavePath}---")
    saveCalibrationParams(args.resultsSavePath, stereoRmse, leftCameraMatrix, leftDistortionCoeffs, rightCameraMatrix, rightDistortionCoeffs, R, T, E, F)
    
    # print(f"Loading calibration results from folder {args.resultsSavePath}")
    # calibrationParams = loadCalibrationResults(args.resultsSavePath)
    

if __name__ == '__main__':
    main()
