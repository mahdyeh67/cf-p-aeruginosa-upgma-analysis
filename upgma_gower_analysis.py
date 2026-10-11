"""
UPGMA Gower-Distance Cluster Analysis of P. aeruginosa Strains (Final)
======================================================================
Corrected computational workflow:
  - Gower-type mixed-variable distance (NOT Euclidean)
  - NaN-aware pairwise mean (missing values excluded per pair)
  - UPGMA / average linkage on precomputed Gower distance
  - Silhouette analysis k=2-6 using same Gower matrix
  - Clean heatmap (black dendrogram, no cluster markers)

Data correction: 62F3 pigment = Pyomelanin (not Pyocianin)

Author: Mahdyeh Neghabi Hajiagha
Date: 2026-10-07
License: MIT
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

# =============================================================================
# Configuration
# =============================================================================
DEFAULT_INPUT = "input_data.csv"
DEFAULT_OUTPUT_DIR = "output"

ANTIBIOTIC_COLS = ["LEV", "CIP", "OFX", "IPM", "AN", "TOB",
                   "GM", "CAZ", "PTZ", "AZT", "MEN", "FEP"]
BINARY_COLS = ["exoS", "exoU", "Pro", "AP", "EX", "ST", "Gen"]
NOMINAL_COLS = ["Pig"]
BF_COL = "BF"
AGE_COL = "Age"
ALL_VARS = ANTIBIOTIC_COLS + BINARY_COLS + NOMINAL_COLS + [BF_COL, AGE_COL]

BF_MAP = {"Negative": 0, "Weak": 1, "Medium": 2, "Moderate": 2, "Strong": 3}

# Color schemes
ABX_COLORS = {0: "#2ca02c", 1: "#ffdd44", 2: "#d62728"}
CAT_PAL = {
    "exoU":  {"Positive": "#1f77b4", "Negative": "#e0e0e0"},
    "exoS":  {"Positive": "#1f77b4", "Negative": "#e0e0e0"},
    "Pro":   {"Positive": "#00897b", "Negative": "#e0e0e0"},
    "BF":    {"Strong": "#b30000", "Medium": "#fd8d3c", "Moderate": "#fd8d3c",
              "Weak": "#fdd49e", "Negative": "#e0e0e0"},
    "AP":    {"Mucoid": "#762a83", "non-Mucoid": "#c7eae5"},
    "Pig":   {"Pyoverdin": "#1b7837", "Pyocianin": "#2166ac",
              "Pyomelanin": "#543005", "No pigment": "#f5f5f5"},
    "EX":    {"Exacerbation": "#d73027", "non-Exacerbation": "#1a9850"},
    "ST":    {"S": "#1f77b4", "T": "#ff7f0e"},
    "Gen":   {"F": "#e377c2", "M": "#17becf"},
}
AGE_CMAP = plt.cm.YlOrRd
ALL_GROUPS = [("Antibiotics", ANTIBIOTIC_COLS)] + [
    ("Virulence genes", ["exoU", "exoS"]), ("Protease", ["Pro"]),
    ("Biofilm", ["BF"]), ("Mucoidity", ["AP"]), ("Pigment", ["Pig"]),
    ("Disease phase", ["EX"]), ("Sample type", ["ST"]),
    ("Gender", ["Gen"]), ("Age", ["Age"]),
]
GROUP_ABBR = {"Antibiotics": "Antibiotics", "Virulence genes": "Virulence",
    "Protease": "Protease", "Biofilm": "Biofilm", "Mucoidity": "Mucoid",
    "Pigment": "Pigment", "Disease phase": "Disease",
    "Sample type": "Sample", "Gender": "Gender", "Age": "Age"}
ALL_FEATURES_VIZ = [c for _, cols in ALL_GROUPS for c in cols]


def load_data(input_csv):
    """Load CSV. Convert antibiotics to numeric (NaN preserved, NOT filled with 0)."""
    df = pd.read_csv(input_csv).dropna(how="all").reset_index(drop=True)
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    for col in ANTIBIOTIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # NaN stays NaN
    return df


def report_missing(df):
    """Report missing values per variable."""
    print("=" * 60)
    print("MISSING-VALUE REPORT")
    print("=" * 60)
    total = 0
    for col in ALL_VARS:
        nm = int(df[col].isna().sum())
        total += nm
        print(f"  {col:8s}: {nm} missing {'— OK' if nm == 0 else '— WARNING'}")
    print(f"\nTotal missing: {total}")
    if total == 0:
        print("Dataset has zero missing values. Results unaffected by NaN handling.")
    return total


def compute_gower_distance(df):
    """
    Gower-type mixed-variable distance (NaN-aware).
    D(i,j) = sum(valid variable distances) / count(valid variables)
    """
    n = len(df)
    strain_names = df["Strain"].tolist()
    bf_codes = df[BF_COL].map(BF_MAP)
    age_vals = df[AGE_COL].astype(float)
    age_min = float(df[AGE_COL].min())
    age_max = float(df[AGE_COL].max())
    age_range = age_max - age_min if age_max > age_min else 1.0

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            contributions = []
            # A. Antibiotics (ordinal, |xi-xj|/2)
            for col in ANTIBIOTIC_COLS:
                vi, vj = df.loc[i, col], df.loc[j, col]
                if pd.notna(vi) and pd.notna(vj):
                    contributions.append(abs(float(vi) - float(vj)) / 2.0)
            # B. Age (continuous, |ai-aj|/range)
            ai, aj = age_vals[i], age_vals[j]
            if pd.notna(ai) and pd.notna(aj):
                contributions.append(abs(float(ai) - float(aj)) / age_range)
            # C. Binary (0 if same, 1 if different)
            for col in BINARY_COLS:
                vi, vj = df.loc[i, col], df.loc[j, col]
                if pd.notna(vi) and pd.notna(vj):
                    contributions.append(0.0 if vi == vj else 1.0)
            # D. BF (ordinal, |bi-bj|/3)
            bi, bj = bf_codes.iloc[i], bf_codes.iloc[j]
            if pd.notna(bi) and pd.notna(bj):
                contributions.append(abs(float(bi) - float(bj)) / 3.0)
            # E. Nominal (0 if same, 1 if different)
            for col in NOMINAL_COLS:
                vi, vj = df.loc[i, col], df.loc[j, col]
                if pd.notna(vi) and pd.notna(vj):
                    contributions.append(0.0 if vi == vj else 1.0)
            # Gower = sum / count of valid variables
            if len(contributions) == 0:
                raise ValueError(f"No valid variables for pair ({i}, {j})")
            d = sum(contributions) / len(contributions)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix, strain_names


def run_upgma(dist_matrix, k_values=[2, 3, 4, 5, 6]):
    """UPGMA (average linkage) on precomputed Gower distance + silhouette."""
    dist_condensed = squareform(dist_matrix, checks=False)
    strain_link = linkage(dist_condensed, method="average")
    strain_order = dendrogram(strain_link, no_plot=True)["leaves"]

    sil_results = []
    assignments = {}
    for k in k_values:
        cl = fcluster(strain_link, t=k, criterion="maxclust")
        score = silhouette_score(dist_matrix, cl, metric="precomputed")
        sizes = sorted(pd.Series(cl).value_counts().values, reverse=True)
        # Relabel by dendrogram position
        oc = cl[strain_order]; seen = []
        for c in oc:
            cid = int(c)
            if cid not in seen: seen.append(cid)
        remap = {old: new for new, old in enumerate(seen, start=1)}
        cl_r = [remap[int(c)] for c in cl]
        assignments[k] = cl_r
        sil_results.append({"k": k, "silhouette": round(score, 4), "sizes": sizes})

    return strain_link, strain_order, sil_results, assignments


def render_heatmap(df, strain_link, strain_order, output_png):
    """Render clean heatmap (black dendrogram, no cluster markers)."""
    n = len(df)
    n_features = len(ALL_FEATURES_VIZ)
    df_display = df.fillna(0)  # for display only
    strain_names = df["Strain"].tolist()

    rgba = np.zeros((n, n_features, 4))
    for j, feat in enumerate(ALL_FEATURES_VIZ):
        if feat == AGE_COL:
            vals = df_display[AGE_COL].astype(float).values
            norm = mcolors.Normalize(vmin=float(vals.min()), vmax=float(vals.max()))
            for i, v in enumerate(vals): rgba[i, j] = AGE_CMAP(norm(v))
        elif feat in ANTIBIOTIC_COLS:
            for i, v in enumerate(df_display[feat].values):
                rgba[i, j] = mcolors.to_rgba(ABX_COLORS.get(int(v), "#888888"))
        else:
            palette = CAT_PAL.get(feat, {})
            for i, v in enumerate(df_display[feat].values):
                rgba[i, j] = mcolors.to_rgba(palette.get(str(v), "#888888"))

    rgba_ordered = rgba[strain_order]
    ordered_strains = [strain_names[i] for i in strain_order]
    group_bounds = []
    pos = 0
    for gname, cols in ALL_GROUPS:
        group_bounds.append((pos, pos + len(cols), gname)); pos += len(cols)

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(28, 18), dpi=120, facecolor="white")
    gs = GridSpec(2, 3, width_ratios=[0.06, 1.0, 0.26],
        height_ratios=[0.06, 1.0], wspace=0.025, hspace=0.02,
        left=0.025, right=0.975, top=0.93, bottom=0.10)

    # Black dendrogram
    ax_dendro = fig.add_subplot(gs[1, 0])
    dendrogram(strain_link, orientation="left", labels=strain_names, ax=ax_dendro,
        color_threshold=0, above_threshold_color="black",
        link_color_func=lambda k: "black", leaf_font_size=7)
    ax_dendro.axis("off")
    ax_dendro.set_title("UPGMA\nDendrogram", fontsize=11, fontweight="bold", pad=10)

    # Top group bars
    ax_groups = fig.add_subplot(gs[0, 1])
    ax_groups.set_xlim(0, n_features); ax_groups.set_ylim(0, 1); ax_groups.axis("off")
    gc = plt.cm.tab20(np.linspace(0, 1, len(ALL_GROUPS)))
    for idx, (start, end, gname) in enumerate(group_bounds):
        ax_groups.add_patch(Rectangle((start, 0.05), end - start, 0.9,
            facecolor=gc[idx], edgecolor="white", linewidth=1.5, alpha=0.75))
        label = GROUP_ABBR.get(gname, gname)
        fs = max(7.5, min(11, 7.5 + (end - start) * 0.5))
        ax_groups.text((start + end) / 2, 0.5, label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color="black")

    # Heatmap (clean — no cluster markers)
    ax_heat = fig.add_subplot(gs[1, 1])
    ax_heat.imshow(rgba_ordered, aspect="auto", interpolation="nearest")
    ax_heat.set_yticks(np.arange(n)); ax_heat.set_yticklabels(ordered_strains, fontsize=7)
    ax_heat.yaxis.tick_right(); ax_heat.yaxis.set_label_position("right")
    ax_heat.tick_params(axis="y", length=0, pad=3)
    ax_heat.set_ylabel("$\\mathit{P.\\ aeruginosa}$ Strains (UPGMA-clustered)",
        fontsize=11, fontweight="bold", labelpad=8)
    display_features = ["Muc" if f == "AP" else f for f in ALL_FEATURES_VIZ]
    ax_heat.set_xticks(np.arange(n_features))
    ax_heat.set_xticklabels(display_features, rotation=90, fontsize=9)
    ax_heat.xaxis.tick_bottom(); ax_heat.xaxis.set_label_position("bottom")
    ax_heat.tick_params(axis="x", length=0, pad=4)
    ax_heat.set_xlabel("Phenotypic / Genotypic / Antibiotic Resistance Features (grouped)",
        fontsize=11, fontweight="bold", labelpad=12)
    for (start, end, gname) in group_bounds[:-1]:
        ax_heat.axvline(x=end - 0.5, color="black", linewidth=2.0)
    for spine in ax_heat.spines.values():
        spine.set_visible(True); spine.set_linewidth(1.5); spine.set_color("black")
    ax_heat.set_xticks(np.arange(-0.5, n_features, 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=0.4)
    ax_heat.tick_params(which="minor", length=0)

    # Legends
    ax_leg = fig.add_subplot(gs[1, 2])
    ax_leg.axis("off"); ax_leg.set_xlim(0, 1); ax_leg.set_ylim(0, 1)
    ax_leg.text(0.18, 1.005, "Color Legends by Feature Group",
        transform=ax_leg.transAxes, fontsize=11, fontweight="bold",
        va="bottom", ha="left", color="#222222")
    total_items = sum(4 if g == "Antibiotics" else (3 if g == "Age" else
        sum(len(CAT_PAL[f]) + 1 for f in cols)) for g, cols in ALL_GROUPS)
    lh = max(0.0085, min((0.97 - 0.01) / (total_items + 4), 0.0160))
    ibg = lh * 0.45; TX = SX = 0.18; SW = 0.085; SH = lh * 0.80
    LX = SX + SW + 0.02; y = 0.97
    def dt(text):
        nonlocal y; ax_leg.text(TX, y, text, fontsize=8.5, fontweight="bold",
            transform=ax_leg.transAxes, va="top", ha="left", color="#222222"); y -= lh * 1.05
    def ds(label, color):
        nonlocal y; ax_leg.add_patch(Rectangle((SX, y - SH), SW, SH,
            transform=ax_leg.transAxes, facecolor=color, edgecolor="black", linewidth=0.4))
        ax_leg.text(LX, y - SH / 2, label, fontsize=7.5, transform=ax_leg.transAxes,
            va="center", ha="left", clip_on=True); y -= lh
    dt("Antibiotics (0/1/2)")
    for code in [0, 1, 2]:
        ds(["Sensitive (0)", "Intermediate (1)", "Resistant (2)"][code], ABX_COLORS[code])
    y -= ibg
    dt(f"Age ({int(df_display['Age'].min())}-{int(df_display['Age'].max())} years)")
    gh = SH * 1.5
    ax_leg.imshow(np.linspace(0, 1, 200).reshape(1, -1), aspect="auto", cmap=AGE_CMAP,
        extent=[SX, SX + 0.55, y - gh, y], transform=ax_leg.transAxes)
    ax_leg.add_patch(Rectangle((SX, y - gh), 0.55, gh, transform=ax_leg.transAxes,
        fill=False, edgecolor="black", linewidth=0.4))
    ax_leg.text(SX, y - gh - 0.005, str(int(df_display["Age"].min())), fontsize=7,
        transform=ax_leg.transAxes, va="top", ha="left")
    ax_leg.text(SX + 0.55, y - gh - 0.005, str(int(df_display["Age"].max())), fontsize=7,
        transform=ax_leg.transAxes, va="top", ha="right")
    y -= gh + lh * 1.2 + ibg
    for gname, cols in ALL_GROUPS[1:]:
        if gname == "Age": continue
        if len(cols) == 1:
            feat = cols[0]; dt(f"{feat}  ({gname})")
            for cat, color in CAT_PAL[feat].items(): ds(cat, color)
            y -= ibg
        else:
            feat = cols[0]; dt(f"{gname}  ({', '.join(cols)})")
            for cat, color in CAT_PAL[feat].items(): ds(cat, color)
            y -= ibg

    fig.text(0.4360, 0.965,
        "UPGMA Grouped Heatmap of $\\mathit{P.\\ aeruginosa}$ Strains",
        ha="center", va="bottom", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.015,
        "Rows = strains (UPGMA clustered, names on right)  |  "
        "Columns = features in groups  |  Color legends on the right",
        ha="center", fontsize=8.5, style="italic", color="#444")
    fig.savefig(output_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="UPGMA Gower-distance cluster analysis for P. aeruginosa")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", "-o", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading: {args.input}")
    df = load_data(args.input)
    print(f"  {len(df)} strains x {df.shape[1]} columns")

    report_missing(df)

    print("\nComputing Gower distance...")
    dist_matrix, strain_names = compute_gower_distance(df)
    print(f"  Shape: {dist_matrix.shape}")
    print(f"  Symmetric: {np.allclose(dist_matrix, dist_matrix.T)}")
    print(f"  Diagonal=0: {np.allclose(np.diag(dist_matrix), 0)}")
    print(f"  Range: [{dist_matrix[dist_matrix > 0].min():.6f}, {dist_matrix.max():.6f}]")

    dist_csv = os.path.join(args.output_dir, "gower_distance_matrix_56x56.csv")
    pd.DataFrame(dist_matrix, index=strain_names, columns=strain_names).to_csv(dist_csv)
    print(f"  Saved: {dist_csv}")

    print("\nRunning UPGMA (average linkage)...")
    strain_link, strain_order, sil_results, assignments = run_upgma(dist_matrix)
    print(f"  Root distance: {strain_link[-1, 2]:.6f}")

    print("\nSilhouette scores:")
    for r in sil_results:
        print(f"  k={r['k']}: {r['silhouette']:.4f}, sizes={r['sizes']}")

    png_path = os.path.join(args.output_dir, "heatmap_final_k4.png")
    print(f"\nRendering heatmap: {png_path}")
    render_heatmap(df, strain_link, strain_order, png_path)

    print("\n=== DONE ===")
    print(f"  PNG:  {png_path}")
    print(f"  CSV:  {dist_csv}")


if __name__ == "__main__":
    main()
