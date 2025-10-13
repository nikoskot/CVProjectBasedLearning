import cv2 as cv
import os
import numpy as np
from cameraCalibration import singleCameraCalibration, loadCalibrationImages, showImagesInGrid

def stereoCameraCalibration(leftImages, rightImages, nCornersPerRow=9, nCornersPerColumn=6, refineCorners=True):
    
    lK, lRs, lTs, lDistortionCoeffs = singleCameraCalibration(leftImages, nCornersPerRow, nCornersPerColumn, refineCorners)
    rK, rRs, rTs, rDistortionCoeffs = singleCameraCalibration(rightImages, nCornersPerRow, nCornersPerColumn, refineCorners)

    print(f"Left Camera Matrix: \n{lK}")
    print(f"Left Camera Distortion coefficients: \n{lDistortionCoeffs}")
    
    print(f"Right Camera Matrix: \n{rK}")
    print(f"Right Camera Distortion coefficients: \n{rDistortionCoeffs}")
    
    
if __name__ == '__main__':

    leftImages, rightImages = loadCalibrationImages("all")
    showImagesInGrid(leftImages)
    showImagesInGrid(rightImages)

    stereoCameraCalibration(leftImages, rightImages, 9, 6, True)
    