import argparse
from TibetanImageOrientation.Utils import create_dir, get_file_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-o", "--file_ext", type=str, required=False, default="jpg")
    parser.add_argument("-b", "--binarize", choices=["yes", "no"], required=False, default="yes")