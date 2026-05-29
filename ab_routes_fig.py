import importlib
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import warehouse_model
importlib.reload(warehouse_model)
from warehouse_model import WarehouseModel


NUM_ORDERS, ITEMS_PER_ORDER, ORDER_SEED, LAM = 10_000, 15, 42, 5.0
CAT_COLORS = {
    "AX": "#1a7f37", "AY": "#3fb950", "AZ": "#7ee787",
    "BX": "#e3b341", "BY": "#f0c674", "BZ": "#f7d9a0",
    "CX": "#f85149", "CY": "#ff7b72", "CZ": "#ffa198",
}


def make_wm():
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


def draw(ax, model, path, title):
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
    if path:
        xs = [p[1] for p in path]; ys = [p[0] for p in path]
        ax.plot(xs, ys, color="#ff00ff", lw=1.6, alpha=0.8, zorder=4)
    dr, dc = model.door
    ax.text(dc, dr, "D", ha="center", va="center",
            fontsize=11, color="white", fontweight="bold", zorder=10)
    ax.set_title(title, fontsize=10, pad=6)


def main():
    wm, orders = make_wm()

    wm.aisle_balanced_slotting(orders, rebuild=True, verbose=False)

    typ = [it for it, _ in Counter(it for o in orders for it in o).most_common(ITEMS_PER_ORDER)]
    o0 = orders[0]

    p_o0_bfs   = wm._route_order_silent(o0)
    p_o0_ca    = wm._route_order_congestion_silent(o0, lam=LAM)
    p_typ_bfs  = wm._route_order_silent(typ)
    p_typ_ca   = wm._route_order_congestion_silent(typ, lam=LAM)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    draw(axes[0, 0], wm, p_o0_bfs,
         f"$o_0$ — AB + BFS: {len(p_o0_bfs)} steps")
    draw(axes[0, 1], wm, p_o0_ca,
         f"$o_0$ — AB + Dijkstra-NN ($\\lambda=5$): {len(p_o0_ca)} steps")
    draw(axes[1, 0], wm, p_typ_bfs,
         f"Typical — AB + BFS: {len(p_typ_bfs)} steps")
    draw(axes[1, 1], wm, p_typ_ca,
         f"Typical — AB + Dijkstra-NN ($\\lambda=5$): {len(p_typ_ca)} steps")
    fig.suptitle("Routes on Aisle-Balanced warehouse: BFS vs Dijkstra-NN",
                 fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    out = "final/fig_ab_routes.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
