import matplotlib
matplotlib.use("Agg")
import importlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

import warehouse_model
importlib.reload(warehouse_model)
from warehouse_model import WarehouseModel


NUM_ORDERS, ITEMS_PER_ORDER, ORDER_SEED = 10_000, 15, 42


def fresh_wm():
    wm = WarehouseModel(rows=52, cols=52, shelf_length=4, shelf_gap=1,
                        shelf_rows_height=2, aisle_width=2, margin=1,
                        shelf_capacity=1, seed=6, door=(23, 0))
    wm.generate(); wm.build_graph()
    wm.abc_classify(csv_path="abc_xyz_dataset.csv",
                    a_pct=0.20, b_pct=0.30, x_cv=0.10, y_cv=0.25)
    wm.assign_items_to_shelves()
    o = wm.generate_orders(num_orders=NUM_ORDERS, items_per_order=ITEMS_PER_ORDER,
                            order_seed=ORDER_SEED, weight_col="Total_Annual_Units", skew=1.0)
    wm.build_heatmap(o)
    return wm, o


configs = []
wm_b, o = fresh_wm()
configs.append(("Baseline ABC$\\times$XYZ", wm_b))

wm_dwc, o = fresh_wm()
wm_dwc.dwc_slotting(o, dist_tol=5.0, max_swaps=None, rebuild=True, verbose=False)
configs.append(("DWC ($\\Delta=5$)", wm_dwc))

wm_hzr, o = fresh_wm()
wm_hzr.hot_zone_relegation(o, hot_pct=0.15, rebuild=True, verbose=False)
configs.append(("Hot-Zone Relegation", wm_hzr))

wm_taps, o = fresh_wm()
wm_taps.taps_slotting(o, alpha=3.0, rebuild=True, verbose=False)
configs.append(("TAPS ($\\alpha=3$)", wm_taps))

wm_ab, o = fresh_wm()
wm_ab.aisle_balanced_slotting(o, rebuild=True, verbose=False)
configs.append(("Aisle-Balanced", wm_ab))

wm_aahms, o = fresh_wm()
wm_aahms.affinity_aware_slotting(o, top_k_pairs=150, max_swaps=400,
                                  congestion_tol=0.10, rebuild=True, verbose=False)
configs.append(("AAHMS ($K=150$)", wm_aahms))

vmax = max(float(c[1].cell_counts.max()) for c in configs)

fig, axes = plt.subplots(2, 3, figsize=(20, 14))
for ax, (title, model) in zip(axes.ravel(), configs):
    disp = model.cell_counts.astype(float)
    mask = np.ma.masked_where(model.grid != 0, disp)
    bg = np.where(model.grid == 1, 0.4, np.where(model.grid == 2, 0.8, 0.05))
    ax.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest", alpha=0.4)
    im = ax.imshow(mask, cmap="YlOrRd",
                   norm=PowerNorm(gamma=0.4, vmin=0, vmax=vmax),
                   alpha=0.9, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, model.grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, model.grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#333", linewidth=0.2)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    gv = model._gini_coefficient(model.cell_counts)
    ax.set_title(f"{title}\nGini={gv:.4f}", fontsize=10, pad=6)
    dr, dc = model.door
    ax.text(dc, dr, "D", ha="center", va="center",
            fontsize=10, color="white", fontweight="bold")

fig.suptitle("Traffic heatmaps: Baseline + 5 slotting strategies",
             fontsize=13, fontweight="bold", y=0.995)
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.12, 0.015, 0.76])
fig.colorbar(im, cax=cbar_ax).set_label(f"Visits (max={int(vmax):,})", fontsize=10)
out = "final/fig_heatmap_comparison_style2.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}, vmax={int(vmax)}")
