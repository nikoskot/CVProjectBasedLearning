import cv2 as cv
import numpy as np
import os
from itertools import groupby
from pathlib import Path

# Data exploration for UA-DETRAC-DATASET-10K.v1 dataset downloaded from Roboflow. This is not the original version of the UA-DETRAC dataset.
def main():
    numClasses = 4
    classesNames = ['truck', 'car', 'van', 'bus']
    classesColors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
    datasetPath = 'D:\\Datasets\\UA-DETRAC-DATASET-10K.v1-2024-11-14-3-44pm.yolov8\\train'
    imageHeight, imageWidth = 640, 640

    imagesFiles = os.listdir(os.path.join(datasetPath, 'images'))
    annotationsFiles = os.listdir(os.path.join(datasetPath, 'labels'))

    imagesFilesBySequence = [list(j) for i, j in groupby(imagesFiles, lambda x:x[:10])]

    for sequence in imagesFilesBySequence:
        for i, framePath in enumerate(sequence):
            
            # Read frame
            frame = cv.imread(os.path.join(datasetPath, 'images', framePath), cv.IMREAD_COLOR)
            
            # Read bbox annotations
            annotations = readAnnotations(os.path.join(datasetPath, 'labels', Path(framePath).stem + '.txt'))
            # Draw annotaion on frame
            for ann in annotations:
                unnormalizedAnnotation = [int(ann[0]), int(ann[1]*imageWidth), int(ann[2]*imageHeight), int(ann[3]*imageWidth), int(ann[4]*imageHeight)]
                boundingBoxUpperLeft = (unnormalizedAnnotation[1] - (unnormalizedAnnotation[3] // 2), unnormalizedAnnotation[2] - (unnormalizedAnnotation[4] // 2))
                boundingBoxLowerRight = (unnormalizedAnnotation[1] + (unnormalizedAnnotation[3] // 2), unnormalizedAnnotation[2] + (unnormalizedAnnotation[4] // 2))
                frame = cv.rectangle(frame, boundingBoxUpperLeft, boundingBoxLowerRight, classesColors[unnormalizedAnnotation[0]])
            
            cv.imshow(framePath[:10], frame)
            key = cv.waitKey(100) & 0xFF
            
            if key == ord('q'):  # q to quit
                cv.destroyAllWindows()
                return

            elif key == 32:  # SPACE key (ASCII 32) move to next sequence
                break
        
        cv.destroyAllWindows()
        
def readAnnotations(filePath):
    annotations = []
    with open(filePath) as f:
        lines = f.readlines()
        for l in lines:
            annotations.append([float(x) for x in l.split()])
    
    return annotations
            
    
if __name__ == '__main__':
    main()