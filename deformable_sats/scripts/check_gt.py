
import numpy as np
import json
from pathlib import Path

def check_gt():
    gt_dir = Path("sats/preprocessing/gt_output_v1")
    meta_path = gt_dir / "ecomesh_d10_z1_test1_gt_meta.json"
    npy_path = gt_dir / "ecomesh_d10_z1_test1_targets.npy"
    
    if not meta_path.exists() or not npy_path.exists():
        print("GT files not found")
        return

    with open(meta_path) as f:
        meta = json.load(f)
    
    print(f"Metadata: {meta['trial_id']}")
    print(f"Shape: {meta['gt_shape']}")
    print(f"Active rows: {meta['n_active_rows']}")
    print(f"Positive Fz: {meta['n_positive_fz']}")
    print(f"Negative Fz: {meta['n_negative_fz']}")

    # Load a few samples
    targets = np.load(npy_path, mmap_mode='r')
    
    # Find an active row
    # We can't easily find it without loading everything, but we can sample
    found = False
    for i in range(0, len(targets), 1000):
        if np.max(targets[i]) > 0:
            print(f"Found active sample at index {i}")
            sample = targets[i]
            # Check where the peak is
            max_idx = np.unravel_index(np.argmax(sample), sample.shape)
            print(f"Peak at {max_idx}")
            
            # Read the merged CSV to see (x, y) at this row
            # Note: generate_gt.py might have dropped rows, so index might not match 1:1 if drop_offgrid is true
            # But let's assume it matches for now or just look at the map symmetry
            found = True
            break
    
    if not found:
        print("No active sample found in sampling")

if __name__ == "__main__":
    check_gt()
