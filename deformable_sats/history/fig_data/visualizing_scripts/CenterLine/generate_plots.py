import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ======================================================
# Configuration
# ======================================================

# fig_data 루트 기준 상대 경로(구 Windows 절대경로 하드코딩 제거)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "fig2_material_ablation"))

CENTERLINE_DATA = {
    "ECO20": os.path.join(BASE_DIR, "CenterLine", "eco20", "d10", "20260529_test2"),
    "EcoMesh": os.path.join(BASE_DIR, "CenterLine", "eco20 + mesh", "d10", "20260530_test1"),
    "ECO50": os.path.join(BASE_DIR, "CenterLine", "eco50", "d10", "20260530_test1"),
}

OUT_DIR = os.path.join(BASE_DIR, "Analysis_Results")
os.makedirs(OUT_DIR, exist_ok=True)

SKIN_COLS = [f"Skin{i}" for i in range(1, 17)]

# Updated Sensor Coordinates from user (x, y)
SENSOR_COORDS = {
    1: [-9.75, -9.75], 2: [-3.25, -9.75], 3: [3.25, -9.75], 4: [9.75, -9.75],
    5: [-9.75, -3.25], 6: [-3.25, -3.25], 7: [3.25, -3.25], 8: [9.75, -3.25],
    9: [-9.75, 3.25], 10: [-3.25, 3.25], 11: [3.25, 3.25], 12: [9.75, 3.25],
    13: [-9.75, 9.75], 14: [-3.75, 9.75], 15: [3.25, 9.75], 16: [9.75, 9.75]
}

# ======================================================
# Utility Functions
# ======================================================

def load_and_sync(data_dir, label):
    print(f"Processing {label} from {data_dir} ...")
    due_path = os.path.join(data_dir, "due_data.csv")
    ether_path = os.path.join(data_dir, "ethermotion_data.csv")
    
    due_df = pd.read_csv(due_path)
    ether_df = pd.read_csv(ether_path)
    
    # 1. Sync Time to Position
    # Scale Y to mm (assuming 10,000 units = 1mm)
    ether_df["Pos_raw"] = ether_df["Y"] / 10000.0
    
    # Average frames within each burst
    burst_avg = due_df.groupby("burst_index")[["time_s"] + SKIN_COLS].mean().reset_index()
    
    # 2. Response Calculation (Relative Change %)
    # Baseline: Initial Average (first 2 seconds)
    baseline_mask = burst_avg["time_s"] < 2.0
    baseline = burst_avg[baseline_mask][SKIN_COLS].mean() if baseline_mask.any() else burst_avg[SKIN_COLS].iloc[:10].mean()
    
    # Formula: - (current - baseline) / baseline * 100 
    # (Flipping sign to show pressure as positive response)
    burst_avg[SKIN_COLS] = - (burst_avg[SKIN_COLS] - baseline) / baseline * 100
    
    # Sync robot position to bursts
    burst_avg["Pos_raw"] = np.interp(burst_avg["time_s"], ether_df["time_s"], ether_df["Pos_raw"])
    
    # 3. Robust Alignment using physical Y-coordinates
    # We find where S13, S14, S15, S16 (Y=9.75) peak in Pos_raw
    top_row = [13, 14, 15, 16]
    peak_pos_raws = []
    for s_id in top_row:
        col = f"Skin{s_id}"
        peak_idx = burst_avg[col].idxmax()
        peak_pos_raws.append(burst_avg.loc[peak_idx, "Pos_raw"])
    
    # physical_y = 9.75
    offset = np.mean(peak_pos_raws) - 9.75
    burst_avg["Pos_mm"] = burst_avg["Pos_raw"] - offset
    print(f"  {label} Alignment Offset: {offset:.2f} mm")
    
    # 4. Handle multiple passes (Average by Position)
    burst_avg["Pos_bin"] = (burst_avg["Pos_mm"] * 10).round() / 10
    final_df = burst_avg.groupby("Pos_bin")[SKIN_COLS].mean().reset_index()
    final_df.rename(columns={"Pos_bin": "Pos_mm"}, inplace=True)
    
    return final_df

def plot_heatmap(data_dict, output_path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    x_min, x_max = -12, 12
    x_bins = np.linspace(x_min, x_max, 400)
    
    # Calculate GLOBAL max across ALL materials for consistent comparison
    all_max = max([df[SKIN_COLS].max().max() for df in data_dict.values()])
    # Use 10% of the overall maximum as a noise floor for "Black" areas
    noise_floor = all_max * 0.1 
    
    for i, (label, df) in enumerate(data_dict.items()):
        df_sorted = df.sort_values("Pos_mm")
        heatmap_data = np.zeros((16, len(x_bins)))
        
        for s_idx, skin in enumerate(SKIN_COLS):
            s_data = df_sorted[skin].values
            # Absolute noise threshold: anything below 10% of the strongest signal is noise
            s_data[s_data < noise_floor] = 0
            heatmap_data[s_idx, :] = np.interp(x_bins, df_sorted["Pos_mm"], s_data)
            
        im = axes[i].imshow(heatmap_data, aspect='auto', cmap="magma", interpolation='nearest',
                             extent=[x_bins[0], x_bins[-1], 16.5, 0.5], vmin=0, vmax=all_max)
        
        fig.colorbar(im, ax=axes[i], label='Relative Change (%)')
        axes[i].set_title(f"Figure A: {label} Response Pattern (Fixed Scale Max: {all_max:.1f}%)")
        axes[i].set_ylabel("Sensor Index")
        axes[i].set_yticks(range(1, 17))
        axes[i].invert_yaxis()

    axes[-1].set_xlabel("Indentation Position (mm)")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")

def plot_curves(data_dict, output_path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
    colors = plt.cm.tab20(np.linspace(0, 1, 16))
    
    # Calculate global max for shared Y-axis comparison
    all_max = max([df[SKIN_COLS].max().max() for df in data_dict.values()])

    for i, (label, df) in enumerate(data_dict.items()):
        df_sorted = df.sort_values("Pos_mm")
        
        for s_idx, skin in enumerate(SKIN_COLS):
            axes[i].plot(df_sorted["Pos_mm"], df_sorted[skin], label=skin, color=colors[s_idx], alpha=0.9, linewidth=1.5)
        
        axes[i].set_title(f"Figure B: {label} Sensor Response Curves")
        axes[i].set_ylabel("Relative Change (%)")
        axes[i].set_xlim(-12, 12)
        axes[i].set_ylim(0, all_max * 1.1)
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend(bbox_to_anchor=(1.02, 0.5), loc='center left', ncol=2, fontsize='x-small')

    axes[-1].set_xlabel("Indentation Position (mm)")
    plt.tight_layout(rect=[0, 0, 0.88, 0.98])
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")

def calculate_metrics(data_dict):
    results = []
    coords = np.array([SENSOR_COORDS[i] for i in range(1, 17)])

    # FIXED Threshold for activation (% change)
    FIXED_THRESHOLD = 5.0 

    for label, df in data_dict.items():
        valid_frames = df[(df["Pos_mm"] >= -10) & (df["Pos_mm"] <= 10)].copy()
        
        if valid_frames.empty:
            results.append({"Material": label, "Active Sensors": 0, "Spread (mm)": 0, "Entropy": 0})
            continue

        active_counts = (valid_frames[SKIN_COLS] > FIXED_THRESHOLD).sum(axis=1)
        active_avg = active_counts[active_counts > 0].mean() if (active_counts > 0).any() else 0
        
        sigmas = []
        entropies = []
        for _, row in valid_frames.iterrows():
            w = np.clip(row[SKIN_COLS].values, 1e-9, None)
            total_w = w.sum()
            
            if total_w > FIXED_THRESHOLD * 2:
                w_norm = w / total_w
                mu = np.sum(w_norm[:, np.newaxis] * coords, axis=0)
                dist_sq = np.sum((coords - mu)**2, axis=1)
                sigmas.append(np.sqrt(np.sum(w_norm * dist_sq)))
                entropies.append(-np.sum(w_norm * np.log(w_norm)))
            
        results.append({
            "Material": label,
            "Active Sensors": active_avg,
            "Spread (mm)": np.mean(sigmas) if sigmas else 0,
            "Entropy": np.mean(entropies) if entropies else 0
        })
        
    return pd.DataFrame(results)

def plot_metrics(metrics_df, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ["Avg Active Sensors", "Spatial Spread (mm)", "Response Entropy"]
    cols = ["Active Sensors", "Spread (mm)", "Entropy"]
    materials = metrics_df["Material"].values
    colors = ['#5da5da', '#faa43a', '#60bd68']
    
    for i, col in enumerate(cols):
        values = metrics_df[col].values
        axes[i].bar(materials, values, color=colors)
        axes[i].set_title(titles[i], fontweight='bold')
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.05*max(values) if max(values)>0 else 0.1, f"{v:.2f}", ha='center')
        
    plt.suptitle("Figure D: Quantitative Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    synced_data = {}
    for label, path in CENTERLINE_DATA.items():
        synced_data[label] = load_and_sync(path, label)
        
    plot_heatmap(synced_data, os.path.join(OUT_DIR, "Figure_A_Heatmaps.png"))
    plot_curves(synced_data, os.path.join(OUT_DIR, "Figure_B_Curves.png"))
    
    metrics_df = calculate_metrics(synced_data)
    metrics_df.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"), index=False)
    plot_metrics(metrics_df, os.path.join(OUT_DIR, "Figure_D_Metrics.png"))
    
    print("\nVisualization complete. Check Analysis_Results folder.")
