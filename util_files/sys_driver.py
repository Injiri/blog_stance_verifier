import sys
import os
import re
import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="injiri blog-news factchecker")
    parser.parse_args()

    dir = "sysFeatures"
    for f in os.listdir(dir):
        if re.search('\.npy$', f):
            filename = os.path.join(dir, f)
            os.remove(filename)
    for f in ['hold_out_ids.txt', 'training_ids.txt']:
        filename = os.path.join('splits_data', f)
        if os.path.isfile(filename):
            os.remove(filename)
    print("All cleard")


def versioning():
    if sys.version_info.major < 3:
        sys.stderr.write('Please use Python version 3 and above\n')
        sys.exit(1)
