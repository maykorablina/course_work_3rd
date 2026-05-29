import matplotlib
matplotlib.use("Agg")
import importlib
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import warehouse_model
importlib.reload(warehouse_model)
from warehouse_model import WarehouseModel


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
    wm.build_heatmap(o)
    return wm, o


CATEGORY_RANK = {"AX": 0, "AY": 1, "AZ": 2, "BX": 3, "BY": 4, "BZ": 5,
                 "CX": 6, "CY": 7, "CZ": 8}


def rank_to_color_dim(rank: int):


    t = rank / 8.0
    if t <= 0.5:
        s = t / 0.5
        return [s * 0.45, 0.40, 0.12 * (1 - s)]
    else:
        s = (t - 0.5) / 0.5
        return [0.45, 0.40 * (1 - s), 0.0]


def zone_rgb(wm, path_bfs, path_ca):
    rows_, cols_ = wm.grid.shape
    set_bfs = set(path_bfs); set_ca = set(path_ca)
    only_bfs = set_bfs - set_ca
    only_ca  = set_ca  - set_bfs
    shared   = set_bfs & set_ca
    rgb = np.zeros((rows_, cols_, 3))
    for r in range(rows_):
        for c in range(cols_):
            v = wm.grid[r, c]
            cell = (r, c)
            if v == 2:
                rgb[r, c] = [0.24, 0.43, 0.95]
            elif v == 1:
                if cell in wm.shelf_assignments and wm.shelf_assignments[cell]:
                    cat = wm.shelf_assignments[cell][0]["ABC_XYZ"]
                    rgb[r, c] = rank_to_color_dim(CATEGORY_RANK.get(cat, 4))
                else:
                    rgb[r, c] = [0.28, 0.28, 0.28]
            else:
                if cell in only_bfs:
                    rgb[r, c] = [0.10, 0.70, 0.15]
                elif cell in only_ca:
                    rgb[r, c] = [0.15, 0.45, 0.90]
                elif cell in shared:
                    rgb[r, c] = [0.88, 0.18, 0.10]
                else:
                    rgb[r, c] = [0.07, 0.07, 0.10]
    return rgb, only_bfs, only_ca, shared


def draw_zone(wm, path_bfs, path_ca, title, save_path):
    rows_, cols_ = wm.grid.shape
    rgb, only_bfs, only_ca, shared = zone_rgb(wm, path_bfs, path_ca)
    fig, ax = plt.subplots(figsize=(cols_ / 2.0, rows_ / 2.0))
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, cols_, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows_, 1), minor=True)
    ax.grid(which="minor", color="#1e1e1e", linewidth=0.3)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    dr, dc = wm.door
    ax.text(dc, dr, "D", ha="center", va="center",
            fontsize=12, color="white", fontweight="bold", zorder=10)
    delta = len(path_ca) - len(path_bfs)
    bfs_cells = max(len(set(path_bfs)), 1)
    relief_pct = len(only_bfs) / bfs_cells * 100
    bottle_pct = len(shared)   / bfs_cells * 100
    ax.set_title(
        f"{title}\n"
        f"Baseline = {len(path_bfs)} steps  |  Config A = {len(path_ca)} steps  "
        f"($\\Delta = {delta:+d}$)\n"
        f"Relieved (green) = {len(only_bfs)} cells ({relief_pct:.0f}%)   "
        f"Bottleneck (red) = {len(shared)} cells ({bottle_pct:.0f}%)",
        fontsize=11, pad=10
    )
    legend_handles = [
        mpatches.Patch(facecolor="#1ab31a", label="Relieved by CA (BFS only)"),
        mpatches.Patch(facecolor="#2472d8", label="CA-only detour"),
        mpatches.Patch(facecolor="#d93318", label="Shared bottleneck (both)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=10, framealpha=0.92)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}  "
          f"relieved={len(only_bfs)} ({relief_pct:.0f}%), "
          f"ca-only={len(only_ca)}, "
          f"shared={len(shared)} ({bottle_pct:.0f}%)")


def main():
    wm, orders = make_wm()
    typ = [it for it, _ in Counter(it for o in orders for it in o).most_common(ITEMS_PER_ORDER)]
    o0 = orders[0]

    p_o0_bfs = wm._route_order_silent(o0)
    p_o0_ca  = wm._route_order_congestion_silent(o0, lam=LAM)
    p_typ_bfs = wm._route_order_silent(typ)
    p_typ_ca  = wm._route_order_congestion_silent(typ, lam=LAM)

    draw_zone(wm, p_o0_bfs, p_o0_ca,
              "Order $o_0$ (random) — Zone Analysis",
              "final/fig_zone_order0.png")
    draw_zone(wm, p_typ_bfs, p_typ_ca,
              "Typical Order (top-15 most frequent) — Zone Analysis",
              "final/fig_zone_typical.png")


if __name__ == "__main__":
    main()
