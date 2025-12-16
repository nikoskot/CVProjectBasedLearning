from pathlib import Path
import os
import cv2 as cv
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.core.frame import Frame

class Dataset():
    
    def __init__(self, path):
        p = Path(path)
        
        self.frames = []
        
        if p.exists():
            
            if p.is_file():
                print(f"Path {path} is a file. Will treat it as a video.")
                
                idx = 0
                cap = cv.VideoCapture(path)
                while cap.isOpened():
                    ret, frame = cap.read()
                
                    if not ret:
                        print("Can't receive frame (stream end?). Exiting.")
                        break
                    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                    frame = Frame(gray, idx)
                    self.frames.append(frame)
                    idx += 1
                    
                    if cv.waitKey(1) == ord('q'):
                        break
                
                cap.release()
                cv.destroyAllWindows()
                
                print(f"Created dataset with {idx} frames.")
            
            if p.is_dir():
                print(f"Path {path} is a folder. Will treat it like it contains a sequence of frames.")
                
                contents = list(p.glob("*"))
                idx = 0
                for c in contents:
                    if c.is_file():
                        frame = cv.imread(str(c))
                        
                        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                        
                        frame = Frame(gray, idx)
                        self.frames.append(frame)
                        idx += 1
                
                print(f"Created dataset with {idx} frames.")        
        else:
            print(f"Path {path} does not exist.")

if __name__ == '__main__':
    dataset = Dataset("Calibration\calibrationImages")
    
    dataset = Dataset("D:\Videos\Arc Raiders\Arc Raiders 2025.12.13 - 21.07.24.05.DVR.mp4")