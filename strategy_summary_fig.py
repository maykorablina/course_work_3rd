import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


data = [
    ("Baseline\nABC×XYZ",       "BL",    55.86, 147, 108.58, 361, 0.6697),
    ("DWC\n(Δ=5)",              "DWC",   55.86, 147, 108.58, 361, 0.6697),
    ("Hot-Zone\nRelegation",    "HZR",   28.20, 389,  67.12, 513, 0.5943),
    ("TAPS\n(α=3)",             "TAPS",  50.82, 111, 118.50, 377, 0.6789),
    ("Aisle-\nBalanced",        "AB",    46.25, 107,  83.37, 353, 0.6678),
    ("AAHMS\n(K=150)",          "AAHMS", 46.00, 115,  83.37, 353, 0.6679),
]
labels    = [d[0] for d in data]
shorts    = [d[1] for d in data]
typ_wpc   = [d[2] for d in data]
typ_steps = [d[3] for d in data]
o0_wpc    = [d[4] for d in data]
o0_steps  = [d[5] for d in data]
ginis     = [d[6] for d in data]


COL = {
    "BL":    "#7f7f7f",
    "DWC":   "#d97a7a",
    "HZR":   "#e85c5c",
    "TAPS":  "#4a90d9",
    "AB":    "#27ae60",
    "AAHMS": "#7fbf7f",
}
colors = [COL[s] for s in shorts]

base_typ_wpc, base_typ_steps = typ_wpc[0], typ_steps[0]
base_o0_wpc,  base_o0_steps  = o0_wpc[0],  o0_steps[0]
base_gini = ginis[0]

fig = plt.figure(figsize=(15, 10))
gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32)


def bar_panel(ax, values, baseline, title, ylabel, fmt="{:.1f}", lower_is_better=True):
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=1.0, zorder=3)
    ax.axhline(baseline, color="#444", ls="--", lw=1.2, alpha=0.8, zorder=2,
               label=f"Baseline = {fmt.format(baseline)}")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02,
                fmt.format(v), ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(shorts, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left" if not lower_is_better else "best")
    ax.grid(axis="y", color="#ddd", linewidth=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(values) * 1.18)

ax1 = fig.add_subplot(gs[0, 0])
bar_panel(ax1, typ_wpc, base_typ_wpc, "WPC — Typical order", "WPC (lower is better)")

ax2 = fig.add_subplot(gs[0, 1])
bar_panel(ax2, typ_steps, base_typ_steps, "Route length — Typical order",
          "Steps (lower is better)", fmt="{:.0f}")

ax3 = fig.add_subplot(gs[0, 2])
bar_panel(ax3, ginis, base_gini, "Gini coefficient", "Gini", fmt="{:.3f}")


ax4 = fig.add_subplot(gs[1, 0])
bar_panel(ax4, o0_wpc, base_o0_wpc, r"WPC — Order $o_0$", "WPC (lower is better)")

ax5 = fig.add_subplot(gs[1, 1])
bar_panel(ax5, o0_steps, base_o0_steps, r"Route length — Order $o_0$",
          "Steps (lower is better)", fmt="{:.0f}")


ax6 = fig.add_subplot(gs[1, 2])
for x_v, y_v, lbl, s in zip(typ_steps, typ_wpc, shorts, shorts):
    ax6.scatter(x_v, y_v, s=240, color=COL[s], edgecolor="white", linewidth=1.6,
                zorder=3)
    dx, dy = 6, 0.4
    if s == "BL":     dx, dy = -8, 1.5
    if s == "HZR":    dx, dy = -10, 1.0
    if s == "DWC":    dx, dy = 6, 1.5
    if s == "TAPS":   dx, dy = -22, -1.8
    if s == "AB":     dx, dy = 4, -1.2
    if s == "AAHMS":  dx, dy = 4, 0.8
    ax6.annotate(lbl, (x_v, y_v), xytext=(x_v + dx, y_v + dy),
                 fontsize=9, fontweight="bold")

best_rect = Rectangle((0, 0), base_typ_steps, base_typ_wpc,
                      alpha=0.12, color="#27ae60", zorder=0)
ax6.add_patch(best_rect)
ax6.text(base_typ_steps * 0.5, base_typ_wpc * 0.30,
         "Best region\n(better than baseline\non both axes)",
         color="#1e8449", fontsize=8.5, ha="center", va="center",
         fontweight="bold")
ax6.scatter(base_typ_steps, base_typ_wpc, marker="x", s=160, color="#444",
            linewidth=2.5, zorder=4)
ax6.set_xlabel("Route length (steps)", fontsize=10)
ax6.set_ylabel("WPC", fontsize=10)
ax6.set_title("Trade-off: WPC vs Route Length\n(Typical order)", fontsize=11, fontweight="bold")
ax6.grid(alpha=0.4, zorder=0)
ax6.spines["top"].set_visible(False)
ax6.spines["right"].set_visible(False)
ax6.set_xlim(0, max(typ_steps) * 1.08)
ax6.set_ylim(0, max(typ_wpc) * 1.08)

fig.suptitle("Slotting strategy comparison: Baseline + 5 strategies (K=10,000 orders, seed 42)",
             fontsize=13, fontweight="bold", y=0.995)
plt.savefig("final/fig_strategy_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: final/fig_strategy_summary.png")
