import cv2 as cv
import os
import numpy as np

CALIBRATION_IMAGES_FOLDER_NAME = "calibrationImages"

def loadCalibrationImages(group="all"):

    calibrationImagesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), CALIBRATION_IMAGES_FOLDER_NAME)

    if group == "left" or group == "right":
        images = []

        imageFiles = os.listdir(calibrationImagesPath)

        for imgFile in imageFiles:
            if imgFile.startswith(group):
                imgPath = os.path.join(calibrationImagesPath, imgFile)
                images.append(cv.imread(imgPath))
        return images
    
    elif group == "all":
        imageFiles = os.listdir(calibrationImagesPath)
        leftImages, rightImages = [], []

        for imgFile in imageFiles:
            if imgFile.startswith("left"):
                imgPath = os.path.join(calibrationImagesPath, imgFile)
                leftImages.append(cv.imread(imgPath))
            if imgFile.startswith("right"):
                imgPath = os.path.join(calibrationImagesPath, imgFile)
                rightImages.append(cv.imread(imgPath))
        return leftImages, rightImages

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
    # cv.resizeWindow("Grid", 1920, 1080)
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

def reprojection_error(world_pts, image_pts, K, R, t):
    # world_pts Nx3 (or Nx2 with z=0)
    pts_h = np.hstack([world_pts, np.zeros((world_pts.shape[0],1))]) if world_pts.shape[1]==2 else world_pts
    projected = []
    for X in pts_h:
        X_h = X.reshape(3,1)
        x_cam = R @ X_h + t.reshape(3,1)
        x = K @ x_cam
        u = x[0]/x[2]; v = x[1]/x[2]
        projected.append([u.item(), v.item()])
    projected = np.array(projected)
    err = np.linalg.norm(projected - image_pts, axis=1)
    return err.mean(), err.std()

def calibrateSingleCamera():

    worldCoords = np.zeros((7*6,3), np.float32)
    worldCoords[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    imageCoords = []

    images = loadCalibrationImages("left")
    showImagesInGrid(images)

    V = []
    Hs = []

    for img in images:

        ret, corners = cv.findChessboardCorners(img, (7, 6))
        if not ret:
            continue

        cv.drawChessboardCorners(img, (7, 6), corners, ret)
        cv.imshow("Annotated corners", img)
        cv.waitKey(500)
        cv.destroyAllWindows()

        imageCoords.append(corners.reshape(-1, 2))

        # Compute homographies for current image
        H = computeHomography(worldCoords, imageCoords[-1])
        Hs.append(H)

        v12 = np.array([H[0, 0]*H[1, 0], H[0, 0]*H[1, 1] + H[0, 1]*H[1, 0], H[0, 1]*H[1, 1], H[0, 2]*H[1, 0] + H[0, 0]*H[1, 2], H[0, 2]*H[1, 1] + H[0, 1]*H[1, 2], H[0, 2]*H[1, 2]])
        v11 = np.array([H[0, 0]*H[0, 0], H[0, 0]*H[0, 1] + H[0, 1]*H[0, 0], H[0, 1]*H[0, 1], H[0, 2]*H[0, 0] + H[0, 0]*H[0, 2], H[0, 2]*H[0, 1] + H[0, 1]*H[0, 2], H[0, 2]*H[0, 2]])
        v22 = np.array([H[1, 0]*H[1, 0], H[1, 0]*H[1, 1] + H[1, 1]*H[1, 0], H[1, 1]*H[1, 1], H[1, 2]*H[1, 0] + H[1, 0]*H[1, 2], H[1, 2]*H[1, 1] + H[1, 1]*H[1, 2], H[1, 2]*H[1, 2]])

        V.append(v12)
        V.append((v11 - v22))

    V = np.array(V)
    _, _, Vh = np.linalg.svd(V)
    b = Vh[-1]

    v0 = (b[1]*b[3] - b[0]*b[4]) / (b[0]*b[2] - b[1]**2)
    l = b[5] - ((b[3]**2 + v0*(b[1]*b[3] - b[0]*b[4])) / b[0])
    alpha = np.sqrt(l / b[0])
    beta = np.sqrt(l*b[0] / (b[0]*b[2] - b[1]**2))
    c = -b[1]* alpha**2 * beta / l
    u0 = c * v0 / alpha - b[3] * alpha**2 / l

    K = np.array([
        [alpha, c, u0],
        [0, beta, v0],
        [0, 0, 1]
    ])

    print(f"Intrinsic Matrix K: \n {K}")

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
        Rs.append(R)
        ts.append(t)
        print(f"Rotation Matrix of {i}: \n {R}")
        print(f"Translation vector of {i}: \n {t}")

        mean_err, std_err = reprojection_error(worldCoords, imageCoords[i], K, R, t)
        print(f"{i} -> mean reprojection err = {mean_err:.3f}px, std = {std_err:.3f}px")

if __name__ == "__main__":
    calibrateSingleCamera()

    leftImages, rightImages = loadCalibrationImages("all")
    print(f"Loaded {len(leftImages)} left images and {len(rightImages)} right images for calibration.")
    # for l, r in zip(leftImages, rightImages):
    #     cv.imshow("left", l)
    #     cv.imshow("right", r)
    #     cv.waitKey(0)

    leftImages, rightImages = [], []
    leftImages = loadCalibrationImages("left")
    rightImages = loadCalibrationImages("right")
    print(f"Loaded {len(leftImages)} left images and {len(rightImages)} right images for calibration.")
    # for l, r in zip(leftImages, rightImages):
    #     cv.imshow("left", l)
    #     cv.imshow("right", r)
    #     cv.waitKey(0)