import cv2
import pandas as pd
import numpy as np
import pickle
from collections import Counter
import random
from sklearn.model_selection import train_test_split

# Import data
DATA_ROOT = "C:/Users/marko/Desktop/NAIST-UBI-RESEARCH/FSAR_Team_Sports/multisports_data/data/trainval"
MEDATATA_PATH = DATA_ROOT + "/multisports_GT.pkl"

with open(MEDATATA_PATH, "rb") as file:
    data = pickle.load(file)

labels = data["labels"]
gttubes = data["gttubes"]

# Get class counts
class_counts = Counter()
for vid, actions in gttubes.items():
    for label_id, tubes in actions.items():
        class_counts[label_id] += len(tubes)

# First remove the lowest 6 classes cause there isn't enough samples to test 5-shot with 20 query samples 
remove_ids = [cid for cid, count in class_counts.items() if count < 25]
for vid_key, vid_tublets in gttubes.items():
    for lab_id in list(vid_tublets.keys()):
        if lab_id in remove_ids:
            del gttubes[vid_key][lab_id]

    if not gttubes[vid_key]:
        print("here")
        del gttubes[vid_key]

# Adjust the counter
class_counts = Counter()
for vid, actions in gttubes.items():
    for label_id, tubes in actions.items():
        class_counts[label_id] += len(tubes)


# Then use the N lowest count classes from the leftover classes for the test set
N_test = 15
N_val = 10
sorted_by_count = sorted(class_counts.items(), key=lambda item: item[1])

# Make validation and test splits from this, use stratified by sport
test_val_ids = sorted_by_count[:N_test + N_val]
test_val_ids = [item[0] for item in test_val_ids[:N_test + N_val]]
train_ids = [item[0] for item in sorted_by_count[N_test + N_val:]]

# Separate the ids by sport and assign randomly 
sports_for_ids = []
for id in test_val_ids:
    sport = labels[id].split(" ")[0]
    sports_for_ids.append(sport)


# Sample stratified
test_ids = []
val_ids = []
test_ids, val_ids = train_test_split(test_val_ids, test_size=N_val, stratify=sports_for_ids, random_state=42)


# For counting number of test and train samples
train_sample_counter = 0
test_sample_counter = 0
val_sample_counter = 0

# Get all the videos with these classes and put them in the test class and the other videos in the train class
test_videos = []
train_videos = []
val_videos = []

# Delete non set tublets
def delete_non_set_tublets(vid_tublets, allowed_ids):
    """Removes keys from the tublet dict that aren't in the allowed list."""
    for lab_id in list(vid_tublets.keys()):
        if lab_id not in allowed_ids:
            del vid_tublets[lab_id]


for vid_key, vid_tublets in gttubes.items():
    possible_train = True


    in_test = False
    test_cnt = 0
    in_val = False
    val_cnt = 0
    for lab_id in list(vid_tublets.keys()):
        if lab_id in test_ids:
            in_test = True
            test_cnt += 1

        elif lab_id in val_ids:
            in_val = True
            val_cnt += 1

    # if video has validation and test samples we use it for the one that has more instances of 
    if val_cnt < test_cnt:
        test_videos.append(vid_key)
        test_sample_counter += sum(len(tubes) for lab_id, tubes in vid_tublets.items() if lab_id in test_ids)
        delete_non_set_tublets(vid_tublets, test_ids)

    elif val_cnt > test_cnt:
        val_videos.append(vid_key)
        val_sample_counter += sum(len(tubes) for lab_id, tubes in vid_tublets.items() if lab_id in val_ids)
        delete_non_set_tublets(vid_tublets, val_ids)

    # if it's 0 for both assign to the train set
    elif val_cnt == 0 and  test_cnt == 0:
        train_videos.append(vid_key)
        train_sample_counter += sum(len(tubes) for tubes in vid_tublets.values())

    # if it's the same but not 0, decide randomly
    else: 
        if random.random() > 0.5:
            test_videos.append(vid_key)
            test_sample_counter += sum(len(tubes) for lab_id, tubes in vid_tublets.items() if lab_id in test_ids)
            delete_non_set_tublets(vid_tublets, test_ids)
        
        else:
            val_videos.append(vid_key)
            val_sample_counter += sum(len(tubes) for lab_id, tubes in vid_tublets.items() if lab_id in val_ids)
            delete_non_set_tublets(vid_tublets, val_ids)
            

# Count the number of test videos and train videos
print(f"Number of test videos: {len(test_videos)}")
print(f"Number of train videos: {len(train_videos)}")
print(f"Number of validation videos: {len(val_videos)}")

# Number of train and test samples
print(f"Number of train instances: {train_sample_counter}")
print(f"Number of test instances: {test_sample_counter}")
print(f"Number of validation instances: {val_sample_counter}")


# Re-indexing
remaining_ids = sorted(class_counts.keys())
label_map = {old_id: new_id for new_id, old_id in enumerate(remaining_ids)}
processed_gttubes = {}

train_label_map = {old_id: new_id for new_id, old_id in enumerate(train_ids)}
val_label_map = {old_id: new_id for new_id, old_id in enumerate(val_ids)}
test_label_map = {old_id: new_id for new_id, old_id in enumerate(test_ids)}

train_labels_list = [labels[old_id] for old_id in train_ids]
val_labels_list = [labels[old_id] for old_id in val_ids]
test_labels_list = [labels[old_id] for old_id in test_ids]
print(train_labels_list)

for vid_key, vid_tublets in gttubes.items():
    if vid_key in train_videos:
        processed_gttubes[vid_key] = {train_label_map[old_id]: tubes for old_id, tubes in vid_tublets.items() if old_id in train_label_map}
    elif vid_key in val_videos:
        processed_gttubes[vid_key] = {val_label_map[old_id]: tubes for old_id, tubes in vid_tublets.items() if old_id in val_label_map}
    elif vid_key in test_videos:
        processed_gttubes[vid_key] = {test_label_map[old_id]: tubes for old_id, tubes in vid_tublets.items() if old_id in test_label_map}


# Save the splits back into a new pickle file
split_dict = {
    "train_labels": train_labels_list,
    "validation_labels": val_labels_list,
    "test_labels": test_labels_list,
    "train_videos": train_videos,
    "test_videos": test_videos,
    "validation_videos": val_videos,
    "nframes": data["nframes"],
    "resolution": data["resolution"],
    "gttubes": processed_gttubes
}

# Create the ground truth pickle file
with open("multisports_fewshot_GT.pkl", "wb") as f:
    pickle.dump(split_dict, f)
