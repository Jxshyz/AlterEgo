import argparse
from record import record

# Setup an argument parser for control via command line
parser = argparse.ArgumentParser(
    prog="Face Swapping Pipeline",
    description="A machine learning pipeline for face Swapping",
    epilog="Students project",
)

parser.add_argument("mode", choices=["record"])
parser.add_argument("-f", "--folder")

# Parse the arguments from the command line
args = parser.parse_args()

# Switch control flow based on arguments
if args.mode == "record":
    record(args)