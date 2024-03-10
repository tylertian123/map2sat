import PIL.Image
import numpy as np
import os
import glob
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("map_dir", type=str, help="Directory for map images.")
    parser.add_argument("sat_dir", type=str, help="Directory for satellite images.")
    parser.add_argument("--delete", "-d", action="store_true", help="Delete images that need to be cleaned. This cannot be undone!")
    parser.add_argument("--threshold", "-t", type=float, default=3.0, help="Standard deviation threshold for empty image detection.")
    parser.add_argument("--out", "-o", type=str, default=None, help="Write out a list of files to delete.")

    args = parser.parse_args()
    if not args.map_dir.endswith("/"):
        args.map_dir = args.map_dir + "/"
    if not args.sat_dir.endswith("/"):
        args.sat_dir = args.sat_dir + "/"

    # Try to find all the PNGs
    # Try directly under this directory first; if that doesn't work, try all subdirectories
    files = glob.glob(args.map_dir + "*.png")
    if not files:
        files = glob.glob(args.map_dir + "**/*.png")
    if not files:
        parser.error("No image files found!")

    cleaned = 0
    total = 0
    paths = []
    for file in files:
        if not os.path.basename(file).startswith("map_"):
            print(f"Skipping file {file} as it does not match conventions")
        total += 1
        img = PIL.Image.open(file).convert("RGB")
        arr = np.array(img)
        # Calculate standard deviation per-channel
        std = np.std(arr, axis=(0, 1))
        if all(s <= args.threshold for s in std):
            print(f"Empty map image: {file} (stdev: {std})")
            cleaned += 1
            # Super sketchy way to find the corresponding sat image name
            name = file[len(args.map_dir):]
            basename = "sat" + os.path.basename(name)[3:]
            sub_path = os.path.dirname(name)
            if sub_path:
                sat_file = args.sat_dir + sub_path + "/" + basename
            else:
                sat_file = args.sat_dir + basename
            print(f"\tSatellite image: {sat_file}")
            if args.out is not None:
                paths.append(file)
                paths.append(sat_file)
            if args.delete:
                os.remove(file)
                os.remove(sat_file)
    if args.out is not None:
        with open(args.out, "w") as f:
            for line in paths:
                f.write(line + "\n")
    print(f"Total: Cleaned {cleaned} image pairs out of {total} ({cleaned / total * 100:.2f}%)")

if __name__ == "__main__":
    main()
