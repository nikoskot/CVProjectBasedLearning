import argparse
import cv2 as cv
import os
import numpy as np
from scipy.optimize import least_squares
import pathlib
import plotly.graph_objs as pgo
import plotly.offline as pyo
import tqdm

def getParser():
    parser = argparse.ArgumentParser(description="Camera Calibration")
    parser.add_argument("--imagesFolder", type=lambda p: pathlib.Path(p).resolve(), default="Calibration\calibrationImages")
    parser.add_argument("--liveCapture", action="store_true")
    parser.add_argument("--imagesGroup", type=str, choices=["left", "right"], default="left")
    parser.add_argument("--patternRowCorners", type=int, default=9)
    parser.add_argument("--patternColumnCorners", type=int, default=6)
    parser.add_argument("--dontRefineCorners", action="store_true")
    parser.add_argument("--resultsSavePath", type=lambda p: pathlib.Path(p).resolve(), default="Calibration\calibrationResults")
    return parser
    
def loadCalibrationImages(group="all", folderName=None):
    """
    This function is used by external code in order to load the images of the calibration pattern. 
    """
    if not pathlib.Path.exists(folderName):
        print(f"The provided folder {folderName} does not exist.")
        return []
    
    imageFiles = os.listdir(folderName)
    if len(imageFiles) == 0:
        print(f"The provided folder {folderName} exists but is empty.")
        return []
    
    if group == "left" or group == "right":
        images = []
        for imgFile in imageFiles:
            if imgFile.startswith(group):
                imgPath = os.path.join(folderName, imgFile)
                images.append(cv.imread(imgPath))
                
        print(f"Loaded {len(images)} {group} images.")
        return images
    
    elif group == "all":
        leftImages, rightImages = [], []
        for imgFile in imageFiles:
            if imgFile.startswith("left"):
                imgPath = os.path.join(folderName, imgFile)
                leftImages.append(cv.imread(imgPath)) 
                
            if imgFile.startswith("right"):
                imgPath = os.path.join(folderName, imgFile)
                rightImages.append(cv.imread(imgPath))
        print(f"Loaded {len(leftImages)} left and {len(rightImages)} right images.")  
        return leftImages, rightImages

def captureCalibrationImagesFromSingleCamera():
    """
    This function is used to capture images of the calibration pattern using a connected camera. Start taking pictures using the SPACE key. Stop with q.
    """
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        exit()
    
    images = []
    capturing = False
    info = "Waiting to start capturing images."
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = frame
        # Display the resulting frame
        cv.imshow(info, gray)
        
        key = cv.waitKey(100) & 0xFF

        if key == ord('q'):  # q to quit
            print("Quitting capture.")
            break

        elif key == 32:  # SPACE key (ASCII 32)
            capturing = True
            cv.destroyAllWindows()
            info = "Capturing images..."
            continue
        
        if capturing:
            images.append(gray)
            # cv.destroyAllWindows()
            # cv.waitKey(100)
        
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    
    print(f"Captured {len(images)} images.")
    return images

def showImagesInGrid(images):

    rows = np.sqrt(len(images)).astype(int)
    cols = len(images) // rows + (len(images) % rows > 0)

    remaining = rows * cols - len(images)

    grid_rows = []
    for r in range(rows):
        row_imgs = images[r*cols:(r+1)*cols]
        row = np.hstack(row_imgs)
        if r == rows - 1 and remaining > 0:
            blank_img = np.zeros_like(images[0])
            for _ in range(remaining):
                row = np.hstack((row, blank_img))
        grid_rows.append(row)
    grid = np.vstack(grid_rows)

    cv.namedWindow("Grid", cv.WINDOW_NORMAL)
    cv.imshow("Grid", grid)
    cv.waitKey(0)
    cv.destroyAllWindows()

def computeHomography(worldCoords, imageCoords):

    n = worldCoords.shape[0]
    A = []

    for i in range(n):
        X, Y = worldCoords[i, 0], worldCoords[i, 1]
        u, v = imageCoords[i, 0], imageCoords[i, 1]

        A.append([X, Y, 1, 0, 0, 0, -u*X, -u*Y, -u])
        A.append([0, 0, 0, X, Y, 1, -v*X, -v*Y, -v])
    
    A = np.array(A)
    _, _, Vh = np.linalg.svd(A)
    h = Vh[-1]
    H = np.array(h.reshape(3, 3))

    return H / H[2, 2]

def v_from_H(H, i, j):
    # generic builder using columns hi and hj
    hi = H[:, i]
    hj = H[:, j]
    return np.array([
        hi[0]*hj[0],
        hi[0]*hj[1] + hi[1]*hj[0],
        hi[1]*hj[1],
        hi[2]*hj[0] + hi[0]*hj[2],
        hi[2]*hj[1] + hi[1]*hj[2],
        hi[2]*hj[2]
    ])

def manualSingleCameraCacibration(images, worldCoords, imageCoords, CompareWithOpenCV=False):

    V = []
    Hs = []

    for i in range(len(worldCoords)):
        # Compute homographies for current image
        H = computeHomography(worldCoords[i], imageCoords[i])
        if CompareWithOpenCV:
            print(f"Homography (manual) of image {i}: \n {H}")
            print(f"Homography (opencv) of image {i}: \n {cv.findHomography(worldCoords[i][:, :2], imageCoords[i][:, :2])[0]}")
        Hs.append(H)

        v12 = v_from_H(H, 0, 1)
        v11 = v_from_H(H, 0, 0)
        v22 = v_from_H(H, 1, 1)

        V.append(v12)
        V.append((v11 - v22))

    V = np.array(V)
    _, _, Vh = np.linalg.svd(V)
    b = Vh[-1]

    v0 = (b[1]*b[3] - b[0]*b[4]) / (b[0]*b[2] - b[1]**2)
    l = b[5] - ((b[3]**2 + v0*(b[1]*b[3] - b[0]*b[4])) / b[0])  # noqa: E741
    alpha = np.sqrt(l / b[0])
    beta = np.sqrt(l*b[0] / (b[0]*b[2] - b[1]**2))
    c = -b[1]* alpha**2 * beta / l
    u0 = c * v0 / alpha - b[3] * alpha**2 / l

    K = np.array([
        [alpha, c, u0],
        [0, beta, v0],
        [0, 0, 1]
    ])

    Rs = []
    ts = []
    for i in range(len(Hs)):
        h1 = Hs[i][:, 0]
        h2 = Hs[i][:, 1]
        h3 = Hs[i][:, 2]

        lam = 1 / np.linalg.norm(np.linalg.inv(K).dot(h1))

        r1 = lam * np.linalg.inv(K).dot(h1)
        r2 = lam * np.linalg.inv(K).dot(h2)
        r3 = np.cross(r1, r2)
        t = lam * np.linalg.inv(K).dot(h3)
        R = np.column_stack((r1, r2, r3))
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        Rs.append(R)
        ts.append(t)

    opencvK, opencvRs, opencvTs = None, None, None
    print(f"K matrix (manual): \n {K}")
    if CompareWithOpenCV:
        h, w, _ = images[0].shape
        _, opencvK, opencvDist, opencvRs, opencvTs = cv.calibrateCamera(worldCoords, imageCoords, (w, h), None, None)
        print(f"K matrix (opencv): \n {opencvK}")
        for i in range(len(Rs)):
            print(f"Rotation Matrix (manual) of {i}: \n {Rs[i]}")
            print(f"Rotation Matrix (opencv) of {i}: \n {cv.Rodrigues(opencvRs[i])[0]}")
            print(f"Translation vector (manual) of {i}: \n {ts[i]}")
            print(f"Translation vector (opencv) of {i}: \n {opencvTs[i]}")

    result = least_squares(distortionCalculationReprojectionError, 
                           x0=[0 ,0, 0, 0], 
                           args=(worldCoords, imageCoords, K, Rs, ts), 
                           # jac='2-point',
                           method='lm',             # or 'lm' if no bounds
                            verbose=2,
                            # x_scale='jac',            # automatic scaling by Jacobian
                            # loss='huber',           # robust loss to downweight outliers
                            max_nfev=2000,
                            ftol=1e-15,
                            xtol=1e-9,
                            gtol=1e-9)
    distortionCoeffs = result.x
    print(f"Manual {distortionCoeffs}")

    return K, Rs, ts, distortionCoeffs

def opencvSingleCameraCalibration(images, worldCoords, imageCoords):
    '''
    Calibrate a monocular camera using the OpenCV implementation.
    '''
    h, w, _ = images[0].shape
    reprojectionRMSE, cameraMatrix, distortionCoeffs, rotationVecs, translationVecs = cv.calibrateCamera(worldCoords, imageCoords, (w, h), None, None)
    
    return reprojectionRMSE, cameraMatrix, distortionCoeffs, rotationVecs, translationVecs 

# def reprojectionError(world_pts, image_pts, K, R, t):
#     # world_pts Nx3 (or Nx2 with z=0)
#     pts_h = np.hstack([world_pts, np.zeros((world_pts.shape[0],1))]) if world_pts.shape[1]==2 else world_pts
#     projected = []
#     for X in pts_h:
#         X_h = X.reshape(3,1)
#         x_cam = R @ X_h + t.reshape(3,1)
#         x = K @ x_cam
#         u = x[0]/x[2]
#         v = x[1]/x[2]
#         projected.append([u.item(), v.item()])
#     projected = np.array(projected)
#     err = np.linalg.norm(projected - image_pts, axis=1)
#     return err.mean(), err.std()

# def reprojectionError2(worldCoords, imageCoords, K, Rs, Ts, distCoeffs=np.array([])):
#     mean_error = 0
#     for i in range(len(worldCoords)):
#         imgpoints2, _ = cv.projectPoints(worldCoords[i], Rs[i], Ts[i], K, distCoeffs)
#         error = cv.norm(imageCoords[i], imgpoints2.squeeze(), cv.NORM_L2)/len(imgpoints2)
#         mean_error += error
    
#     return mean_error/len(worldCoords)

def calculateMonocularReprojectionRMSE(worldCoords, imageCoords, cameraMatrix, rotationVecs, translationVecs, distortionCoeffs=np.array([])):
    totalError = 0
    totalPoints = 0

    for wrld, img, r, t in zip(worldCoords, imageCoords, rotationVecs, translationVecs):
        projected, _ = cv.projectPoints(wrld, r, t, cameraMatrix, distortionCoeffs)
        projected = projected.reshape(-1, 2)

        err = np.linalg.norm(img - projected, axis=1)
        totalError += np.sum(err**2)
        totalPoints += len(wrld)
    
    rmse = np.sqrt(totalError / totalPoints)
    return rmse

def calculateReprojectionErrorScatterPlot(worldCoords, imageCoords, cameraMatrix, rotationVecs, translationVecs, distortionCoeffs=np.array([])):
    errorsX = []
    errorsY = []
    imgIds = []
    ptsIds = []
    
    for id, (wrld, img, r, t) in enumerate(zip(worldCoords, imageCoords, rotationVecs, translationVecs)):
        projected, _ = cv.projectPoints(wrld, r, t, cameraMatrix, distortionCoeffs)
        projected = projected.reshape(-1, 2)

        errorsX.extend(img.reshape(-1, 2)[:, 0] - projected[:, 0])
        errorsY.extend(img.reshape(-1, 2)[:, 1] - projected[:, 1])
        imgIds.extend([id] * len(img))
        ptsIds.extend(list(range(len(img))))
    
    scatter  = pgo.Scatter(x=errorsX,
                           y=errorsY,
                           mode='markers',
                           marker=dict(size=10, color='royalblue'),
                           text = imgIds,
                           hoverinfo='text'
                           )
    layout = pgo.Layout(
        title='XY Reprojection Error',
        xaxis=dict(
            title='Error X',
            scaleanchor='y',  # Fix aspect ratio 1:1 by linking x axis scale to y axis scale
            scaleratio=1
            ),
        yaxis=dict(title='Error Y')
        )
    fig = pgo.Figure(data=[scatter], layout=layout)
    pyo.plot(fig)

def distortionCalculationReprojectionError(dist_coeffs, worldCoords, imageCoords, K, Rs, Ts):
    errors = []
    k1, k2, p1, p2 = dist_coeffs
    
    for R, t, img_points, world_points in zip(Rs, Ts, imageCoords, worldCoords):
        for X, uv_obs in zip(world_points, img_points):
            # project to camera coords
            Xc = R @ X + t
            x, y = Xc[0]/Xc[2], Xc[1]/Xc[2]   # normalized

            # radial distortion
            r2 = x*x + y*y
            x_rad = x * (1 + k1*r2 + k2*(r2**2))
            y_rad = y * (1 + k1*r2 + k2*(r2**2))

            # tangential distortion
            x_tan = 2*p1*x*y + p2*(r2 + 2*x*x)
            y_tan = p1*(r2 + 2*y*y) + 2*p2*x*y

            x_d = x_rad + x_tan
            y_d = y_rad + y_tan

            # pixel coords
            u_pred = K[0,0]*x_d + K[0,1]*y_d + K[0,2]
            v_pred = K[1,1]*y_d + K[1,2]

            # residual
            u_obs, v_obs = uv_obs
            # errors.append(u_obs - u_pred)
            # errors.append(v_obs - v_pred)
            errors.append(np.linalg.norm(np.array([u_obs, v_obs]) - np.array([u_pred, v_pred])))
    
    return np.array(errors)

def distortionCalculationReprojectionError2(dist_coeffs, worldCoords, imageCoords, K, Rs, Ts):
    errors = []
    for i in range(len(worldCoords)):
        imgpoints2, _ = cv.projectPoints(worldCoords[i], Rs[i], Ts[i], K, dist_coeffs)
        imgpoints2 = imgpoints2.squeeze()
        for obs, pred in zip(imageCoords[i], imgpoints2):
            # errors.append((obs[0] - pred[0])**2)
            # errors.append((obs[1] - pred[1])**2)
            errors.append(np.linalg.norm(obs - pred))
    return np.array(errors)

# def calibrateSingleCamera(images):

#     worldCoordsSingle = np.zeros((N_CORNERS_PER_ROW*N_CORNERS_PER_COL, 3), np.float32)
#     worldCoordsSingle[:, :2] = np.mgrid[0:N_CORNERS_PER_ROW, 0:N_CORNERS_PER_COL].T.reshape(-1, 2)
#     imageCoords = [] 
#     worldCoords = []

#     for i, img in enumerate(images):
#         # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#         ret, corners = cv.findChessboardCorners(img, (N_CORNERS_PER_ROW, N_CORNERS_PER_COL))
#         if not ret:
#             print(f"No chessboard corners found in image {i}")
#             cv.imshow(f"No corners {i}", img)
#             cv.waitKey(1000)
#             cv.destroyAllWindows()
#             continue
        
#         if REFINE_CORNERS_FLAG:
#             corners = cv.cornerSubPix(cv.cvtColor(img, cv.COLOR_BGR2GRAY), corners, (11,11), (-1,-1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
#             # corners = cv.cornerSubPix(img, corners, (11,11), (-1,-1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            
#         cv.drawChessboardCorners(img, (N_CORNERS_PER_ROW, N_CORNERS_PER_COL), corners, ret)
#         cv.imshow(f"Annotated corners {i}", img)
#         cv.waitKey(500)
#         cv.destroyAllWindows()

#         imageCoords.append(corners.reshape(-1, 2))
#         worldCoords.append(worldCoordsSingle)

#     startTime = time.time()
#     K, Rs, Ts, distCoeffs = manualSingleCameraCacibration(images, worldCoords, imageCoords, CompareWithOpenCV=True)
#     endTime = time.time()
#     print(f"Manual calibration took {endTime - startTime:.3f} seconds.")

#     startTime = time.time()
#     opencvRMSE, opencvK, opencvDistCoeffs, opencvRs, opencvTs = opencvSingleCameraCalibration(images, worldCoords, imageCoords)
#     endTime = time.time()
#     print(f"OpenCV calibration took {endTime - startTime:.3f} seconds.")
#     print(f"Opencv {opencvDistCoeffs}")

#     print(f"Reprojection error of manual calibration: \n {calculateMonocularReprojectionRMSE(worldCoords, imageCoords, K, [cv.Rodrigues(R)[0] for R in Rs], Ts, distCoeffs)}")
    
#     print(f"Reprojection error of opencv calibration: \n {calculateMonocularReprojectionRMSE(worldCoords, imageCoords, opencvK, opencvRs, opencvTs, opencvDistCoeffs)}")

def monocularCameraCalibration(images, nCornersPerRow=9, nCornersPerColumn=6, refineCorners=True, savePath=None):
    """
    This function is used by external code in order to calibrate a single camera using as input the images of the calibration pattern.
    """
    worldCoordsSingle = np.zeros((nCornersPerRow*nCornersPerColumn, 3), np.float32)
    worldCoordsSingle[:, :2] = np.mgrid[0:nCornersPerRow, 0:nCornersPerColumn].T.reshape(-1, 2)
    imageCoords = [] 
    worldCoords = []

    for i, img in tqdm.tqdm(enumerate(images)):
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
        
        if savePath:
            imgCopy = img.copy()
            cv.drawChessboardCorners(imgCopy, (nCornersPerRow, nCornersPerColumn), corners, ret)
            cv.imwrite(pathlib.Path.joinpath(savePath, f"annotated{i}.png"), imgCopy)

        imageCoords.append(corners.reshape(-1, 2))
        worldCoords.append(worldCoordsSingle)

    reprojectionRMSE, cameraMatrix, distortionCoeffs, rotationVecs, translationVecs = opencvSingleCameraCalibration(images, worldCoords, imageCoords)

    print(f"Monocular calibration RMSE (pixels): \n{reprojectionRMSE}")
    print(f"Camera matrix: \n{cameraMatrix}")
    print(f"Camera distortion coefficients: \n{distortionCoeffs}")
    print(f"Rotation vectors: \n{rotationVecs}")
    print(f"Translation vectors: \n{translationVecs}")
    
    calculateReprojectionErrorScatterPlot(worldCoords, imageCoords, cameraMatrix, rotationVecs, translationVecs, distortionCoeffs)
    
    return reprojectionRMSE, cameraMatrix, distortionCoeffs, rotationVecs, translationVecs

def saveCalibrationResults(folderPath, rmse, cameraMatrix, distortionCoeffs):
    try:
        fs = cv.FileStorage(pathlib.Path.joinpath(folderPath, "calib.json"), cv.FILE_STORAGE_WRITE)
        fs.write("cameraMatrix", cameraMatrix)
        fs.write("distortionCoeffs", distortionCoeffs)
        fs.write("rmse", rmse)
        fs.release()
    except Exception as e:
        print(f"Could not save calibration results in file {folderPath}\calib.json. \n Exception {e}")

def loadCalibrationResults(folderPath):
    try:
        fs = cv.FileStorage(pathlib.Path.joinpath(folderPath, "calib.json"), cv.FILE_STORAGE_READ)
        cameraMatrix = fs.getNode("cameraMatrix").mat()
        distortionCoeffs = fs.getNode("distortionCoeffs").mat()
        rmse = fs.getNode("rmse").real()
        fs.release()
        return rmse, cameraMatrix, distortionCoeffs
    except Exception as e:
        print(f"Could not load calibration results from file {folderPath}\calib.json. \n Exception {e}")

def main():
    parser = getParser()
    args = parser.parse_args()

    # Create necessary folders/paths
    print("Creating path for calibration results.")
    os.makedirs(args.resultsSavePath, exist_ok=True)
    os.makedirs(pathlib.Path.joinpath(args.resultsSavePath, "annotatedImages"), exist_ok=True)

    print(f"Starting monocular camera calibration with: \n {vars(args)}")
    
    images = []
    if args.liveCapture:
        print("Capturing images from camera.")
        images = captureCalibrationImagesFromSingleCamera()
    else:
        print(f"Loading images from folder {args.imagesFolder}.")
        images = loadCalibrationImages(group=args.imagesGroup, folderName=args.imagesFolder)
    if len(images) == 0:
        print("No images to use. Quitting.")
        return

    # showImagesInGrid(images)

    reprojectionRMSE, cameraMatrix, distortionCoeffs, rotationVecs, translationVecs = monocularCameraCalibration(images=images, nCornersPerRow=args.patternRowCorners, nCornersPerColumn=args.patternColumnCorners, refineCorners=(not args.dontRefineCorners), savePath=pathlib.Path.joinpath(args.resultsSavePath, "annotatedImages"))
    
    print(f"Saving calibration results to folder {args.resultsSavePath}")
    saveCalibrationResults(args.resultsSavePath, reprojectionRMSE, cameraMatrix, distortionCoeffs)
    
    # print(f"Loading calibration results from folder {args.resultsSavePath}")
    # reprojectionRMSE, cameraMatrix, distortionCoeffs = loadCalibrationResults(args.resultsSavePath)
    # print(f"Loaded monocular calibration RMSE: \n{reprojectionRMSE}")
    # print(f"Loaded camera matrix: \n{cameraMatrix}")
    # print(f"Loaded camera distortion coefficients: \n{distortionCoeffs}")

if __name__ == "__main__":
    main()