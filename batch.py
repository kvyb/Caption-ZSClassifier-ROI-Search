import os
import argparse

parser = argparse.ArgumentParser(description="Run main.py for each image in subject-image directory.")
parser.add_argument("--reference", required=True, help="Path to the reference image file.")
args = parser.parse_args()

reference_path = args.reference

filenames = sorted(os.listdir("subject-image"))

for filename in filenames:
    if filename.endswith(".png"):
        subject_path = os.path.join("subject-image", filename)
        print(subject_path)
        os.system("python3 main.py --reference {} --subject {}".format(reference_path, subject_path))