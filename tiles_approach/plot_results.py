#!/usr/bin/env python3
"""
plot_results.py
Generates four figures from sweep_results.csv for the IEEE paper.

Usage:
    python3 scripts/plot_results.py

Output (saved to results/):
    fig1_speedup_vs_threads.png   -- V1 vs V3 speedup per N
    fig2_time_vs_N.png            -- wall time per version at fixed thread count
    fig3_interchange_gain.png     -- single-thread V0 vs V2 (cache effect only)
    fig4_schedule_crossover.png   -- V3 static vs dynamic crossover
"""

import csv, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

CSV_CANDIDATES = [
    "results/sweep_results.csv",
    "results/sweep_all.csv",
]
OUT_DIR     = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ── load data ─────────────────────────────────────────────────
CSV_PATH = None
for candidate in CSV_CANDIDATES:
    if os.path.exists(candidate):
        CSV_PATH = candidate
        break
if CSV_PATH is None:
    raise FileNotFoundError(
        f"Could not find any sweep CSV file. Checked: {', '.join(CSV_CANDIDATES)}"
    )

rows = []
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        rows.append({k: (float(v) if k not in
                         ('schedule','correct_v1','correct_v2','correct_v3','N',
                          'threads','l2_kib') else v)
                     for k, v in row.items()})
        rows[-1]['N']       = int(rows[-1]['N'])
        rows[-1]['threads'] = int(rows[-1]['threads'])
        rows[-1]['l2_kib']  = int(rows[-1]['l2_kib'])

Ns      = sorted(set(r['N']       for r in rows))
threads = sorted(set(r['threads'] for r in rows))

COLORS = {
    'V0': '#333333',
    'V1': '#E07B39',   # orange
    'V2': '#3A86C8',   # blue
    'V3': '#2BAE66',   # green
}
MARKERS = {'V0':'o','V1':'s','V2':'^','V3':'D'}

# ── helper ────────────────────────────────────────────────────
def get(rows, n, t, key):
    for r in rows:
        if r['N']==n and r['threads']==t:
            return r[key]
    return None

# ─────────────────────────────────────────────────────────────
# Figure 1: Speedup vs Thread Count  (one line per N)
# Shows V1 (naive) and V3 (full pipeline) speedup over V0 seq
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

for ax, vkey, vtitle, vcolor in [
    (axes[0], 'speedup_v1', 'V1: Naive Parallel (i,j,k)', COLORS['V1']),
    (axes[1], 'speedup_v3', 'V3: Full Pipeline (i,k,j + chunk + adaptive)',
     COLORS['V3']),
]:
    for N in Ns:
        sp = [get(rows, N, t, vkey) for t in threads
              if get(rows, N, t, vkey) is not None]
        th = [t for t in threads
              if get(rows, N, t, vkey) is not None]
        ax.plot(th, sp, marker='o', label=f'N={N}', linewidth=1.8)

    # ideal speedup line
    ax.plot(threads, threads, 'k--', linewidth=1, alpha=0.4, label='Ideal')
    ax.set_title(vtitle, fontsize=9)
    ax.set_xlabel('Number of Threads', fontsize=9)
    ax.set_ylabel('Speedup over Sequential V0', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.set_major_locator(ticker.FixedLocator(threads))

plt.suptitle('Figure 1: Speedup vs Thread Count', fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig1_speedup_vs_threads.png', dpi=180,
            bbox_inches='tight')
plt.close()
print("Saved fig1_speedup_vs_threads.png")

# ─────────────────────────────────────────────────────────────
# Figure 2: Wall Time vs N  (at max thread count)
# All four versions, to show scaling behaviour
# ─────────────────────────────────────────────────────────────
max_t = max(threads)
fig, ax = plt.subplots(figsize=(6, 4.5))

for vkey, label, color, marker in [
    ('t_v0_avg', 'V0: Sequential i,j,k',          COLORS['V0'], 'o'),
    ('t_v1_avg', 'V1: Naive Parallel i,j,k',       COLORS['V1'], 's'),
    ('t_v2_avg', 'V2: Interchange Only (seq)',      COLORS['V2'], '^'),
    ('t_v3_avg', 'V3: Full Pipeline',               COLORS['V3'], 'D'),
]:
    times = [get(rows, N, max_t, vkey) for N in Ns
             if get(rows, N, max_t, vkey) is not None]
    ns    = [N for N in Ns if get(rows, N, max_t, vkey) is not None]
    if times:
        ax.plot(ns, times, marker=marker, color=color,
                label=label, linewidth=1.8)

ax.set_xlabel('Matrix Dimension N', fontsize=9)
ax.set_ylabel('Wall-Clock Time (s)', fontsize=9)
ax.set_title(f'Figure 2: Execution Time vs Problem Size  '
             f'(threads={max_t})', fontsize=9)
ax.legend(fontsize=8)
ax.grid(True, linestyle='--', alpha=0.4)
ax.xaxis.set_major_locator(ticker.FixedLocator(Ns))
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig2_time_vs_N.png', dpi=180, bbox_inches='tight')
plt.close()
print("Saved fig2_time_vs_N.png")

# ─────────────────────────────────────────────────────────────
# Figure 3: Loop Interchange Cache Effect (single thread)
# V0 vs V2, sequential only — isolates cache benefit
# ─────────────────────────────────────────────────────────────
t1_rows = [r for r in rows if r['threads'] == min(threads)]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

ax = axes[0]
v0_times = [get(rows, N, min(threads), 't_v0_avg') for N in Ns]
v2_times = [get(rows, N, min(threads), 't_v2_avg') for N in Ns]
ax.plot(Ns, v0_times, marker='o', color=COLORS['V0'],
        label='V0: i,j,k (stride-N on B)', linewidth=1.8)
ax.plot(Ns, v2_times, marker='^', color=COLORS['V2'],
        label='V2: i,k,j (stride-1 on all)', linewidth=1.8)
ax.set_xlabel('Matrix Dimension N', fontsize=9)
ax.set_ylabel('Wall-Clock Time (s)', fontsize=9)
ax.set_title('Sequential: V0 vs V2 (Loop Interchange)', fontsize=9)
ax.legend(fontsize=8); ax.grid(True, linestyle='--', alpha=0.4)
ax.xaxis.set_major_locator(ticker.FixedLocator(Ns))

ax = axes[1]
gains = [v0/v2 for v0,v2 in zip(v0_times, v2_times)
         if v0 and v2]
ax.bar([str(N) for N in Ns], gains, color=COLORS['V2'], alpha=0.8,
       edgecolor='black', linewidth=0.7)
ax.axhline(1.0, color='red', linestyle='--', linewidth=1, label='Baseline (1×)')
ax.set_xlabel('Matrix Dimension N', fontsize=9)
ax.set_ylabel('Speedup of V2 over V0 (1 thread)', fontsize=9)
ax.set_title('Cache Speedup from Loop Interchange Alone', fontsize=9)
ax.legend(fontsize=8); ax.grid(True, axis='y', linestyle='--', alpha=0.4)

plt.suptitle('Figure 3: Loop Interchange Cache Performance Gain',
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig3_interchange_gain.png', dpi=180,
            bbox_inches='tight')
plt.close()
print("Saved fig3_interchange_gain.png")

# ─────────────────────────────────────────────────────────────
# Figure 4: V3 vs V1 — Full Pipeline vs Naive at all N × threads
# Heatmap of speedup ratio V3/V1
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

matrix = np.zeros((len(Ns), len(threads)))
for i, N in enumerate(Ns):
    for j, t in enumerate(threads):
        t1 = get(rows, N, t, 't_v1_avg')
        t3 = get(rows, N, t, 't_v3_avg')
        if t1 and t3 and t1 > 0:
            matrix[i, j] = t1 / t3

im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.8, vmax=2.5)
plt.colorbar(im, ax=ax, label='Speedup of V3 over V1')

ax.set_xticks(range(len(threads)))
ax.set_xticklabels([str(t) for t in threads])
ax.set_yticks(range(len(Ns)))
ax.set_yticklabels([str(N) for N in Ns])
ax.set_xlabel('Number of Threads', fontsize=9)
ax.set_ylabel('Matrix Dimension N', fontsize=9)
ax.set_title('Figure 4: V3 Pipeline vs V1 Naive — Speedup Ratio',
             fontsize=9)

for i in range(len(Ns)):
    for j in range(len(threads)):
        ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center',
                fontsize=8, color='black')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig4_pipeline_vs_naive_heatmap.png', dpi=180,
            bbox_inches='tight')
plt.close()
print("Saved fig4_pipeline_vs_naive_heatmap.png")

print("\nAll figures saved to results/")
