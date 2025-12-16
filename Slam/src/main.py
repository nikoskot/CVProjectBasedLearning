import configargparse
import yaml
from pathlib import Path

def getParser():
    parser = configargparse.ArgParser(default_config_files=["Slam\configs\default.yaml"])
    parser.add("--configFile", is_config_file=True, help='config file path')
    # parser = argparse.ArgumentParser(description="Camera Calibration")
    parser.add("--datasetPath", type=lambda p: Path(p).resolve(), default="Slam\data")
    parser.add("--resultsSavePath", type=lambda p: Path(p).resolve(), default="Slam\results")
    parser.add("--cameraParamsFile", type=lambda p: Path(p).resolve(), default="Slam\data\camera\intrinsics.txt")
    return parser

def main():
    pass

if __name__ == "__main__":
    main()