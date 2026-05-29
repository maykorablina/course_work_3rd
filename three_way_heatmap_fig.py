import importlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

import warehouse_model
importlib.reload(warehouse_model)
from warehouse_model import WarehouseModel
from central_aisle_slotting import central_aisle_slotting


NUM_ORDERS, ITEMS_PER_ORDER, ORDER_SEED, LAM = 10_000, 15, 42, 5.0


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
    return wm, o


def build_heatmap_dijkstra(wm, orders, lam):

    grid_rows, grid_cols = wm.grid.shape
    cell_counts = np.zeros((grid_rows, grid_cols), dtype=float)
    edge_counts = {}
    for order in orders:
        path = wm._route_order_congestion_silent(order, lam=lam)
        for cell in path:
            cell_counts[cell[0], cell[1]] += 1.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge = (u, v) if u <= v else (v, u)
            edge_counts[edge] = edge_counts.get(edge, 0.0) + 1.0
    return cell_counts


def draw_heatmap(ax, wm, counts, title):
    rows_, cols_ = wm.grid.shape

    base = np.ones((rows_, cols_, 3))
    for r in range(rows_):
        for c in range(cols_):
            v = wm.grid[r, c]
            if v == 1:
                base[r, c] = [0.78, 0.78, 0.78]
            elif v == 2:
                base[r, c] = [0.24, 0.43, 0.95]
    ax.imshow(base, interpolation="nearest")


    mask = wm.grid != 1
    overlay = np.zeros_like(counts)
    overlay[mask] = counts[mask]
    ax.imshow(overlay, cmap="YlOrRd",
              norm=PowerNorm(gamma=0.4, vmin=0, vmax=VMAX),
              alpha=0.85, interpolation="nearest")

    dr, dc = wm.door
    ax.text(dc, dr, "D", ha="center", va="center",
            fontsize=11, color="white", fontweight="bold", zorder=10)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    ax.set_title(title, fontsize=11, pad=6)


def main():
    global VMAX


    print("=== Baseline + Dijkstra-NN ===")
    wm_bl, orders_bl = make_wm()
    wm_bl.build_heatmap(orders_bl)
    counts_bl = build_heatmap_dijkstra(wm_bl, orders_bl, lam=LAM)


    print("=== CAS + Dijkstra-NN ===")
    wm_cas, orders_cas = make_wm()
    wm_cas.build_heatmap(orders_cas)
    central_aisle_slotting(wm_cas, orders_cas)
    wm_cas.build_heatmap(orders_cas)
    counts_cas = build_heatmap_dijkstra(wm_cas, orders_cas, lam=LAM)


    print("=== AB + Dijkstra-NN ===")
    wm_ab, orders_ab = make_wm()
    wm_ab.aisle_balanced_slotting(orders_ab, rebuild=True, verbose=False)
    wm_ab.build_heatmap(orders_ab)
    counts_ab = build_heatmap_dijkstra(wm_ab, orders_ab, lam=LAM)

    VMAX = max(counts_bl.max(), counts_cas.max(), counts_ab.max())
    print(f"vmax across three heatmaps: {int(VMAX)}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    draw_heatmap(axes[0], wm_bl, counts_bl,
                 f"Baseline ABC×XYZ + Dijkstra-NN\n(max cell visits: {int(counts_bl.max())})")
    draw_heatmap(axes[1], wm_cas, counts_cas,
                 f"CAS + Dijkstra-NN\n(max cell visits: {int(counts_cas.max())})")
    draw_heatmap(axes[2], wm_ab, counts_ab,
                 f"Aisle-Balanced + Dijkstra-NN\n(max cell visits: {int(counts_ab.max())})")
    fig.suptitle("Traffic heatmaps under Dijkstra-NN routing ($\\lambda=5$, $K=10{,}000$ orders)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = "final/fig_three_way_heatmap.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}, vmax={int(VMAX)}")


if __name__ == "__main__":
    main()
