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

def calibrateSingleCamera():

    worldCoords = np.zeros((8*6,3), np.float32)
    worldCoords[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
    imageCoords = []

    images = loadCalibrationImages("left")
    showImagesInGrid(images)

    for img in images:

        ret, corners = cv.findChessboardCorners(img, (8, 6))
        if ret:
            cv.drawChessboardCorners(img, (8, 6), corners, ret)
            cv.imshow("Annotated corners", img)
            cv.waitKey(500)
            cv.destroyAllWindows()

            imageCoords.append(corners)
    


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