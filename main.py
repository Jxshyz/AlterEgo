import argparse
from Detector import detect
from CamDetect import detect_with_camera

# Setup an argument parser for control via command line
parser = argparse.ArgumentParser(
    prog="Face Swapping Pipeline",
    description="A machine learning pipeline for face swapping",
    epilog="Students project",
)

parser.add_argument(
    "mode",
    choices=["detect", "detect_with_camera"],
    help="Choose 'detect' to process without camera or 'detect_with_camera' to capture live data.",
)
# Parse the arguments from the command line
args = parser.parse_args()

# Switch control flow based on arguments
if args.mode == "detect":
    detect()
elif args.mode == "detect_with_camera":
    detect_with_camera()
