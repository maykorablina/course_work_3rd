import matplotlib
matplotlib.use("Agg")
import importlib
from collections import Counter
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

import warehouse_model
importlib.reload(warehouse_model)
from warehouse_model import WarehouseModel
from central_aisle_slotting import central_aisle_slotting


NUM_ORDERS, ITEMS_PER_ORDER, ORDER_SEED = 10_000, 15, 42

CAT_COLORS = {
    "AX": "#1a7f37", "AY": "#3fb950", "AZ": "#7ee787",
    "BX": "#e3b341", "BY": "#f0c674", "BZ": "#f7d9a0",
    "CX": "#f85149", "CY": "#ff7b72", "CZ": "#ffa198",
}


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


def draw_layout(ax, model, title, order_items=None):


    rows_, cols_ = model.grid.shape
    rgb = np.zeros((rows_, cols_, 3))
    for r in range(rows_):
        for c in range(cols_):
            v = model.grid[r, c]
            if v == 1:
                cell = (r, c)
                if cell in model.shelf_assignments and model.shelf_assignments[cell]:
                    cat = model.shelf_assignments[cell][0]["ABC_XYZ"]
                    rgb[r, c] = mcolors.to_rgb(CAT_COLORS.get(cat, "#888888"))
                else:
                    rgb[r, c] = [0.55, 0.55, 0.55]
            elif v == 2:
                rgb[r, c] = [0.24, 0.43, 0.95]
            else:
                rgb[r, c] = [0.10, 0.10, 0.14]
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, cols_, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows_, 1), minor=True)
    ax.grid(which="minor", color="#222", linewidth=0.3)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    dr, dc = model.door
    ax.text(dc, dr, "D", ha="center", va="center",
            fontsize=12, color="white", fontweight="bold", zorder=10)

    if order_items is not None:
        order_set = set(order_items)
        target_cells = []
        for cell, items in model.shelf_assignments.items():
            for item in items:
                if item["Item_ID"] in order_set:
                    target_cells.append(cell)
                    break
        if target_cells:
            ys = [c[0] for c in target_cells]
            xs = [c[1] for c in target_cells]
            ax.scatter(xs, ys, s=160, facecolor="#00e5ff", edgecolor="black",
                       linewidth=1.2, zorder=5,
                       label=f"Order items ({len(target_cells)} shelves)")
            ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.set_title(title, fontsize=11, pad=8)


def category_legend(fig):
    handles = [mpatches.Patch(facecolor=CAT_COLORS[c], edgecolor="#444",
                              label=c, linewidth=0.5)
               for c in ["AX", "AY", "AZ", "BX", "BY", "BZ", "CX", "CY", "CZ"]]
    fig.legend(handles=handles, loc="lower center", ncol=9,
               fontsize=10, framealpha=0.92, bbox_to_anchor=(0.5, -0.03),
               title="SKU Category (A→C = high→low demand, X→Z = low→high variability)",
               title_fontsize=9)


def main():

    print("Building AB model...")
    wm_ab, orders_ab = fresh_wm()
    wm_ab.aisle_balanced_slotting(orders_ab, rebuild=True, verbose=False)

    print("Building CAS model...")
    wm_cas, orders_cas = fresh_wm()
    central_aisle_slotting(wm_cas, orders_cas)


    print("Building baseline model for order derivation...")
    wm_base, orders_base = fresh_wm()
    typ = [it for it, _ in Counter(it for o in orders_base for it in o).most_common(ITEMS_PER_ORDER)]
    o0 = orders_base[0]


    fig, axes = plt.subplots(1, 2, figsize=(22, 11))
    draw_layout(axes[0], wm_ab,  "Aisle-Balanced Slotting (AB)")
    draw_layout(axes[1], wm_cas, "Central-Aisle Slotting (CAS)")
    category_legend(fig)
    fig.suptitle("Slotting layout comparison: Aisle-Balanced vs Central-Aisle",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out1 = "final/fig_ab_vs_cas_layout.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out1}")


    fig, axes = plt.subplots(1, 2, figsize=(22, 11))
    draw_layout(axes[0], wm_ab,  "AB — item positions for order $o_0$",  order_items=o0)
    draw_layout(axes[1], wm_cas, "CAS — item positions for order $o_0$", order_items=o0)
    category_legend(fig)
    fig.suptitle("Item positions for the random order $o_0$ (cyan dots)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out2 = "final/fig_ab_vs_cas_order0.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out2}")


    fig, axes = plt.subplots(1, 2, figsize=(22, 11))
    draw_layout(axes[0], wm_ab,  "AB — item positions for typical order",  order_items=typ)
    draw_layout(axes[1], wm_cas, "CAS — item positions for typical order", order_items=typ)
    category_legend(fig)
    fig.suptitle("Item positions for the typical order (top-15 most frequent SKUs, cyan dots)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out3 = "final/fig_ab_vs_cas_typical.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out3}")


if __name__ == "__main__":
    main()
