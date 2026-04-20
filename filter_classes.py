import pickle
from collections import Counter
import pandas as pd

# Standard options for full terminal display
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def analyze_multisports_pure_test(pkl_path, n_rare=16):
    # 1. Load the data
    try:
        with open(pkl_path, "rb") as file:
            data = pickle.load(file)
    except FileNotFoundError:
        print(f"Error: Could not find {pkl_path}")
        return

    labels = data["labels"]
    gttubes = data["gttubes"]

    # 2. Calculate Original Counts
    original_counts = Counter()
    for vid, actions in gttubes.items():
        for label_idx, tubes in actions.items():
            original_counts[label_idx] += len(tubes)

    # 3. Identify the N most infrequent classes (Novel Set)
    rare_indices = set(sorted(original_counts, key=original_counts.get)[:n_rare])

    # 4. Partition Videos
    # We still need to find which videos contain the rare classes to remove them from Train
    test_video_ids = set()
    for vid, actions in gttubes.items():
        if any(idx in rare_indices for idx in actions.keys()):
            test_video_ids.add(vid)

    train_video_ids = set(gttubes.keys()) - test_video_ids

    # 5. Count Tubes in Each Split (STRICT FILTERING)
    # Train Set: Total tubes in the "Base" videos
    train_tube_count = 0
    remaining_counts = Counter()
    for vid in train_video_ids:
        for label_idx, tubes in gttubes[vid].items():
            count = len(tubes)
            train_tube_count += count
            remaining_counts[label_idx] += count

    # Test Set: ONLY count tubes belonging to the RARE classes
    test_tube_count = 0
    for vid in test_video_ids:
        for label_idx, tubes in gttubes[vid].items():
            if label_idx in rare_indices:  # <--- THIS IS THE "PURITY" FILTER
                test_tube_count += len(tubes)

    # 6. Build the Complete Comparison Table
    comparison_list = []
    for idx, name in enumerate(labels):
        orig = original_counts[idx]
        rem = remaining_counts[idx]
        loss_pct = ((orig - rem) / orig * 100) if orig > 0 else 0
        
        status = "RARE (Novel Set)" if idx in rare_indices else "COMMON (Base Set)"
        loss_type = "Intentional (Novelty)" if idx in rare_indices else "Collateral Damage"
        if loss_pct == 0: loss_type = "No Loss"

        comparison_list.append({
            "ID": idx,
            "Class Name": name,
            "Original": orig,
            "Remaining": rem,
            "Loss %": round(loss_pct, 2),
            "Loss Type": loss_type,
            "Status": status
        })

    df = pd.DataFrame(comparison_list)
    df = df.sort_values(by="Original", ascending=False).reset_index(drop=True)

    # 7. Detailed Printout
    print(f"\n" + "="*95)
    print(f"PURE NOVELTY ANALYSIS (N_RARE={n_rare})")
    print("="*95)
    print(df.to_string(index=False, col_space=12))
    print("="*95)

    # 8. Final Efficiency Stats
    total_orig_tubes = sum(original_counts.values())
    
    print(f"\n--- FINAL DATASET METRICS ---")
    print(f"Total Videos in Dataset:    {len(gttubes)}")
    print(f"Total Action Tubes (Orig):  {total_orig_tubes}")
    
    print(f"\n--- VIDEO SPLIT SUMMARY ---")
    print(f"TRAIN SET (Base):   {len(train_video_ids)} videos")
    print(f"TEST SET (Novel):    {len(test_video_ids)} videos")
    
    print(f"\n--- TUBE SPLIT SUMMARY (STRICT) ---")
    print(f"TRAIN TUBES (All classes in base vids): {train_tube_count}")
    print(f"TEST TUBES  (Only Rare Classes):        {test_tube_count}")
    print(f"Systematic Data Loss:                   {total_orig_tubes - (train_tube_count + test_tube_count)} tubes")
    print(f" (Common actions inside test videos were discarded to ensure purity)")
    print(f"---------------------------------------------\n")
    
    return df

# --- EXECUTION ---
PATH = "multisports_data/data/trainval/multisports_GT.pkl"
N_RARE = 21

full_report_df = analyze_multisports_pure_test(PATH, n_rare=N_RARE)