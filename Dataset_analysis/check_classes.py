import pickle 
import pandas as pd
from collections import Counter

with open("multisports_data/data/trainval/multisports_GT.pkl", "rb") as file:
    data = pickle.load(file)

labels = data["labels"]
print(f"--- Dataset Overview ---")
print(f"Total Classes: {len(labels)}")
print(f"Keys: {list(data.keys())}")

# 1. Fix the generator print
train_counts = [len(x) for x in data["train_videos"]]
test_counts = [len(x) for x in data["test_videos"]]

print(f"\nVideos per split (sub-lists):")
print(f"Train counts: {train_counts} (Total: {sum(train_counts)})")
print(f"Test counts:  {test_counts} (Total: {sum(test_counts)})")

# Pick a random video and its first action tube
vid = list(data['gttubes'].keys())[0]
label_idx = list(data['gttubes'][vid].keys())[0]
tube = data['gttubes'][vid][label_idx][0] # The first tube for this action

print(f"Video: {vid} | Action: {data['labels'][label_idx]}")
print(f"Number of frames in this action: {len(tube)}")
print(f"Starts at frame: {tube[0, 0]} | Ends at frame: {tube[-1, 0]}")
print(f"First box coordinates: {tube[0, 1:]}")


labels = data["labels"]
gttubes = data["gttubes"]

# 1. Collect all label indices from every tube in every video
all_instances = []
for vid_name, actions in gttubes.items():
    for label_idx, tubes_list in actions.items():
        # One label_idx can have multiple tubes (different people doing the same action)
        for _ in range(len(tubes_list)):
            all_instances.append(labels[label_idx])

# 2. Count them up
counts = Counter(all_instances)

# 3. Create a nice DataFrame for viewing
df_counts = pd.DataFrame(counts.items(), columns=['Action Class', 'Total Tubes'])
df_counts = df_counts.sort_values(by='Total Tubes', ascending=False).reset_index(drop=True)

# Print results
print(df_counts[-20:])