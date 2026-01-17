import pandas as pd

# Read the CSV file
df = pd.read_csv("short_videos2.csv")

# Get unique paths
unique_paths = df['path'].unique()

# Remove leading/trailing whitespace and quotes
unique_paths = [path.strip().strip("'") for path in unique_paths]

print(f"Total unique paths: {len(unique_paths)}")
print("\nUnique paths:")
for path in sorted(unique_paths):
    print(path)

# Optionally save to a new CSV
unique_df = pd.DataFrame({'path': unique_paths})
unique_df.to_csv("unique_paths2.csv", index=False)
print("\nUnique paths saved to unique_paths.csv")