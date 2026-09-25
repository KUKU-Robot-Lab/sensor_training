
import numpy as np
import json
import pandas as pd
from pathlib import Path

def verify_v2():
    gt_dir = Path("sats/preprocessing/gt_output_v2")
    trial_id = "ecomesh_d10_z1_test1"
    meta_path = gt_dir / f"{trial_id}_gt_meta.json"
    npy_path = gt_dir / f"{trial_id}_targets.npy"
    csv_path = Path("raw_data/ecomesh/d10/z_1.0mm/test1/ecomesh_d10_z1.0_test1_merged.csv")
    
    if not meta_path.exists() or not npy_path.exists() or not csv_path.exists():
        print(f"Files missing for {trial_id}")
        return

    with open(meta_path) as f:
        meta = json.load(f)
    
    print(f"--- Verification for {trial_id} ---")
    print(f"FZ Mode: {meta['fz_mode']}")
    print(f"Active rows: {meta['n_active_rows']} / Total rows: {meta['n_total_rows']}")

    # Load targets (mmap)
    targets = np.load(npy_path, mmap_mode='r')
    
    # Load CSV to get coordinates
    df = pd.read_csv(csv_path)
    
    # Filter CSV rows using same logic as generate_gt.py to match targets index
    # grid_x/y used in generate_gt.py: np.linspace(-9.75, 9.75, 40)
    grid_vals = np.linspace(-9.75, 9.75, 40)
    grid_step = 0.5
    grid_tol = 0.05
    
    x_raw = df["x_mm"].to_numpy()
    y_raw = df["y_mm"].to_numpy()
    
    i_cx = np.rint((x_raw - grid_vals[0]) / grid_step).astype(int)
    j_cy = np.rint((y_raw - grid_vals[0]) / grid_step).astype(int)
    
    x_snap = grid_vals[0] + i_cx * grid_step
    y_snap = grid_vals[0] + j_cy * grid_step
    
    on_grid = (i_cx >= 0) & (i_cx < 40) & (j_cy >= 0) & (j_cy < 40) & \
              (np.abs(x_raw - x_snap) <= grid_tol) & (np.abs(y_raw - y_snap) <= grid_tol)
    
    df_grid = df[on_grid].reset_index(drop=True)
    
    if len(df_grid) != targets.shape[0]:
        print(f"Mismatch in row count! CSV grid rows: {len(df_grid)}, Target rows: {targets.shape[0]}")
        return

    # Check orientation for a few samples
    # Pick indices where we expect significant pressure
    test_indices = [2000, 10000, 50000]
    for idx in test_indices:
        if idx >= len(df_grid): continue
        
        row = df_grid.iloc[idx]
        target_map = targets[idx]
        
        if np.max(target_map) == 0:
            print(f"Index {idx}: Target map is zero (Fz too small)")
            continue
            
        # Physical coordinates from CSV
        px, py = row["x_mm"], row["y_mm"]
        # Expected grid indices (0-39)
        exp_ix = round((px - (-9.75)) / 0.5)
        exp_iy = round((py - (-9.75)) / 0.5)
        
        # Actual peak index in target map
        # Now target_map is expected to be [y, x]
        peak_iy, peak_ix = np.unravel_index(np.argmax(target_map), target_map.shape)
        
        print(f"Index {idx}:")
        print(f"  Coord: (x={px:.2f}, y={py:.2f})")
        print(f"  Expected Grid Index: (x_idx={exp_ix}, y_iy={exp_iy})")
        print(f"  Actual Peak Index:   (x_idx={peak_ix}, y_iy={peak_iy})")
        
        if exp_ix == peak_ix and exp_iy == peak_iy:
            print("  Result: SUCCESS (Matching OK)")
        else:
            print("  Result: FAIL (Orientation or Index mismatch)")

if __name__ == "__main__":
    verify_v2()
