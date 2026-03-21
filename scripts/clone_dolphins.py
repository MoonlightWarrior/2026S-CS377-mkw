import os
import shutil
import argparse

def copy_dir(source, destination):
    if os.path.exists(destination):
        shutil.rmtree(destination)
    
    shutil.copytree(source, destination)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clones', type=int, default=3)

    args = parser.parse_args()
    clones_count = args.clones

    for i in range(1, clones_count+1):
        copy_dir("dolphin0", f"dolphin{i}")

if __name__ == "__main__":
    main()