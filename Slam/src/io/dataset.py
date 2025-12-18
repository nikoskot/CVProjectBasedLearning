from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import cv2 as cv
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.core.frame import Frame
from src.core.pose import Pose

class Dataset():
    
    def __init__(self, imagesPath):
        p = Path(imagesPath)
        
        self.frames = [] # list of Frame objects
        
        if p.exists():
            
            if p.is_file():
                print(f"Path {imagesPath} is a file. Will treat it as a video.")
                
                idx = 0
                cap = cv.VideoCapture(imagesPath)
                while cap.isOpened():
                    ret, frame = cap.read()
                
                    if not ret:
                        print("Can't receive frame (stream end?). Exiting.")
                        break
                    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                    frame = Frame(gray, idx, cap.get(cv.CAP_PROP_POS_MSEC))
                    self.frames.append(frame)
                    idx += 1
                    
                    if cv.waitKey(1) == ord('q'):
                        break
                
                cap.release()
                cv.destroyAllWindows()
                
                print(f"Created dataset with {idx} frames.")
            
            if p.is_dir():
                print(f"Path {imagesPath} is a folder. Will treat it like it contains a sequence of frames.")
                
                imageFiles = list(p.glob("*"))
                idx = 0
                for imgFile in imageFiles:
                    if imgFile.is_file():
                        frame = cv.imread(str(imgFile))
                        
                        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                        
                        # Try to get timestap from file name. If impossible timestamp is equal to -1 always
                        timestamp = -1
                        try:
                            timestamp = float(Path(imgFile).stem)
                        except ValueError:
                            pass
                        
                        frame = Frame(gray, idx, timestamp)
                        self.frames.append(frame)
                        idx += 1
                
                print(f"Created dataset with {idx} frames.")        
        else:
            print(f"Path {imagesPath} does not exist.")
            
class GroundtruthPosesDataset():
    
    def __init__(self, groundTruthPosesPath, framesDataset):
        
        self.gtPoses = []
        
        if groundTruthPosesPath:
            print("Recieved path ground truth poses file. Loading them.")
            
            idx = 0
            framesDatasetIdx = 0
            try:
                with open(groundTruthPosesPath, 'r') as file:
                    for line in file:
                        if line.startswith('#'):
                            continue
                        
                        parts = line.split(' ')
                        
                        currentGroundTruthPoseTimestamp = float(parts[0])
                        currentFrameTimestamp = framesDataset.frames[framesDatasetIdx].timestamp
                        
                        if (currentGroundTruthPoseTimestamp < currentFrameTimestamp):
                            continue
                        
                        tx = float(parts[1])
                        ty = float(parts[2])
                        tz = float(parts[3])
                        qx = float(parts[4])
                        qy = float(parts[5])
                        qz = float(parts[6])
                        qw = float(parts[7])
                        
                        rotationMatrix = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                        self.gtPoses.append(Pose(idx, rotationMatrix, np.array([tx, ty, tz])))
                        idx += 1
                        framesDatasetIdx += 1

                    print(f"Created ground truth poses dataset with {idx} poses.")
                    
            except Exception as e:
                print(f"Could not load ground truth poses from file {groundTruthPosesPath}. \nException {e}")
                
if __name__ == '__main__':
    dataset = Dataset("D:\\Documents\\Repos\\CVProjectBasedLearning\\Slam\\data\\rgbd_dataset_freiburg1_xyz\\rgb")
    
    # dataset = Dataset("D:\Videos\Arc Raiders\Arc Raiders 2025.12.13 - 21.07.24.05.DVR.mp4")
    
    gtPosesDataset = GroundtruthPosesDataset("D:\\Documents\\Repos\\CVProjectBasedLearning\\Slam\\data\\rgbd_dataset_freiburg1_xyz\\groundtruth.txt", dataset)