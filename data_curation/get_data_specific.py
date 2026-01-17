import json
import csv

with open("video_tags.json") as f:
    ann = json.load(f)

with open("ids_to_path.json") as f:
    ids_to_path = json.load(f)

def mentions_pick_or_grab(text):
    return "pick" in text.lower() or "grab" in text.lower()

def mentions_mug_or_cup(text):
    return "mug" in text.lower() or "cup" in text.lower()

def matches_criteria(instructions):
    texts = [
        instructions.get("language_instruction1", ""),
        instructions.get("language_instruction2", ""),
        instructions.get("language_instruction3", "")
    ]
    return any(
        mentions_pick_or_grab(t) and mentions_mug_or_cup(t)
        for t in texts
    )

mug_eps = [ep_id for ep_id, instructions in ann.items() if matches_criteria(instructions)]
mug_eps = [ep_id for ep_id in mug_eps if ep_id in ids_to_path]

# Replace IDs with paths
mug_paths = ["gs://gresearch/robotics/droid_raw/1.0.1/" + ids_to_path[ep_id] for ep_id in mug_eps]

print("Episode IDs:", mug_eps)
print("Number of episodes with pick/grab and mug/cup:", len(mug_eps))

# Save to CSV
with open("path_to_mug_videos.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["path"])
    for path in mug_paths:
        writer.writerow([path])

print("Paths saved to path_to_mug_videos.csv")