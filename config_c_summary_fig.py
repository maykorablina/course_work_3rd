import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

labels = ["Baseline\n(B+BFS)", "Config A\n(B+CA)", "Config B-CAS\n(CAS+BFS)", "Config C\n(CAS+CA)"]
shorts = ["BL", "A", "B-CAS", "C"]


data = {
    "Baseline":     (147, 55.86, 361, 108.58),
    "Config A":     ( 85, 16.79, 249,  24.89),
    "Config B-CAS": ( 69, 29.56, 369, 109.15),
    "Config C":     ( 83,  9.15, 239,  23.79),
}
typ_steps = [v[0] for v in data.values()]
typ_wpc   = [v[1] for v in data.values()]
o0_steps  = [v[2] for v in data.values()]
o0_wpc    = [v[3] for v in data.values()]


COL = {"BL": "#7f7f7f", "A": "#4a90d9", "B-CAS": "#d68910", "C": "#27ae60"}
colors = [COL[s] for s in shorts]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

def bar(ax, vals, baseline, title, ylabel, fmt="{:.1f}"):
    x = np.arange(len(vals))
    bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=1.0, zorder=3)
    ax.axhline(baseline, color="#444", ls="--", lw=1.2, alpha=0.8, zorder=2,
               label=f"Baseline = {fmt.format(baseline)}")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02,
                fmt.format(v), ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(shorts, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(axis="y", color="#ddd", linewidth=0.5, zorder=0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(vals) * 1.18)

bar(axes[0, 0], typ_steps, typ_steps[0], "Typical order — steps",   "Steps (lower is better)", fmt="{:.0f}")
bar(axes[0, 1], typ_wpc,   typ_wpc[0],   "Typical order — WPC",     "WPC (lower is better)")
bar(axes[1, 0], o0_steps,  o0_steps[0],  "Order $o_0$ — steps",     "Steps (lower is better)", fmt="{:.0f}")
bar(axes[1, 1], o0_wpc,    o0_wpc[0],    "Order $o_0$ — WPC",       "WPC (lower is better)")

fig.suptitle("Four-configuration comparison: 2×2 design closed by Config C",
             fontsize=13, fontweight="bold", y=0.995)
plt.tight_layout()
out = "final/fig_config_c_summary.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
