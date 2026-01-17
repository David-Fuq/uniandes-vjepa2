import csv
import subprocess
import os

# Read paths from CSV
with open("path_to_mug_videos_redux_100_200.csv", "r") as f:
    reader = csv.DictReader(f)
    paths = [row["path"] for row in reader]

# Create destination directory
dest_dir = "./droid_raw_data_mug_fixes/"
os.makedirs(dest_dir, exist_ok=True)

print(f"Found {len(paths)} videos to download")
print(f"Downloading to: {dest_dir}")

# Download each video
for i, path in enumerate(paths, 1):
    print(f"\n[{i}/{len(paths)}] Downloading: {path}")
    
    try:
        cmd = ["gsutil", "-m", "cp", "-r", path, dest_dir]
        subprocess.run(cmd, check=True)
        print(f"Successfully downloaded")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

print("\nDownload complete!")