#!/usr/bin/env python3
"""Overlay rollout joint trajectories on demonstration trajectories for one task."""
import json, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

task = sys.argv[1]; R = Path("/mount/scratch4/omkar/datasets/wave-traj")
root = R / f"lerobot/{task}-state-15hz"
names = json.load(open(root / "meta/info.json"))["features"]["observation.state"]["names"]
tb = pq.read_table(root / "data/chunk-000/file-000.parquet").to_pandas()
demos = [np.stack(g["observation.state"].values) for _, g in tb.groupby("episode_index")]
rolls = [np.load(p) for p in sorted((R / "rollouts" / task).glob("rollout_*.npz"))]
show = {"wave-right": ["arm_r_joint1", "arm_r_joint3", "arm_r_joint4", "arm_l_joint1"],
        "wave-left":  ["arm_l_joint1", "arm_l_joint3", "arm_l_joint4", "arm_r_joint1"],
        "wave-both":  ["arm_l_joint1", "arm_l_joint3", "arm_r_joint1", "arm_r_joint3"]}[task]
fig, axes = plt.subplots(len(show), 1, figsize=(11, 2.4 * len(show)), sharex=True)
for ax, jn in zip(axes, show):
    j = names.index(jn)
    for k, d in enumerate(demos):
        ax.plot(np.arange(len(d)) / 15, d[:, j], color="0.75", lw=1, label="demonstrations" if k == 0 else None)
    for k, r in enumerate(rolls):
        ax.plot(r["times"], r["states"][:, j], lw=1.2, alpha=0.9, label="policy rollouts" if k == 0 else None)
    ax.set_ylabel(f"{jn}\n[rad]"); ax.grid(alpha=0.3)
axes[0].set_title(f"{task}: state-only Diffusion Policy in MuJoCo — {len(rolls)} rollouts from sampled initial states vs {len(demos)} demos")
axes[0].legend(loc="upper right", fontsize=8); axes[-1].set_xlabel("time [s]")
fig.tight_layout(); out = R / "rollouts" / task / f"trajectories_{task}.png"; fig.savefig(out, dpi=110); print(out)
