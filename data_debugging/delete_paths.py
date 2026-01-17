import pandas as pd
import shutil
import os

# Read the CSV file
df = pd.read_csv("unique_paths2.csv")

# Get all paths
paths = df['path'].tolist()

print(f"Found {len(paths)} paths to delete")

deleted_count = 0
error_count = 0

for i, path in enumerate(paths, 1):
    print(f"\n[{i}/{len(paths)}] Deleting: {path}")
    
    try:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"✓ Successfully deleted directory")
                deleted_count += 1
            else:
                os.remove(path)
                print(f"✓ Successfully deleted file")
                deleted_count += 1
        else:
            print(f"⚠ Path does not exist, skipping")
    except PermissionError as e:
        print(f"✗ Permission denied: {e}")
        error_count += 1
    except Exception as e:
        print(f"✗ Error deleting: {e}")
        error_count += 1

print(f"\n{'='*50}")
print(f"Deletion complete!")
print(f"Successfully deleted: {deleted_count}")
print(f"Errors: {error_count}")
print(f"Skipped (not found): {len(paths) - deleted_count - error_count}")