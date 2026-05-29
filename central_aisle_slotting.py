import matplotlib
matplotlib.use("Agg")

import importlib
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import PowerNorm

import warehouse_model
importlib.reload(warehouse_model)
from warehouse_model import WarehouseModel


NUM_ORDERS, ITEMS_PER_ORDER, ORDER_SEED = 10_000, 15, 42
CENTRAL_AISLE_ROWS = {23, 24}


def make_wm():
    wm = WarehouseModel(
        rows=52, cols=52, shelf_length=4, shelf_gap=1,
        shelf_rows_height=2, aisle_width=2, margin=1,
        shelf_capacity=1, seed=6, door=(23, 0),
    )
    wm.generate()
    wm.build_graph()
    wm.abc_classify(csv_path="abc_xyz_dataset.csv",
                    a_pct=0.20, b_pct=0.30, x_cv=0.10, y_cv=0.25)
    wm.assign_items_to_shelves()
    orders = wm.generate_orders(num_orders=NUM_ORDERS,
                                items_per_order=ITEMS_PER_ORDER,
                                order_seed=ORDER_SEED,
                                weight_col="Total_Annual_Units", skew=1.0)
    wm.build_heatmap(orders)
    return wm, orders


def compute_wpc(path, ec):
    wpc = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge = (u, v) if u <= v else (v, u)
        wpc += ec.get(edge, 0.0)
    return wpc


def central_aisle_slotting(wm: WarehouseModel, orders):

    grid = wm.grid
    rows_, cols_ = grid.shape
    dist_corr = wm.corridor_distances()

    def get_access(cell):
        r, c = cell
        best, min_d = None, float("inf")
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < rows_ and 0 <= cc < cols_ and grid[rr, cc] in (0, 2):
                if dist_corr[rr, cc] < min_d:
                    min_d = dist_corr[rr, cc]
                    best = (rr, cc)
        return best

    def bfs_d(cell):
        ac = get_access(cell)
        return float(dist_corr[ac[0], ac[1]]) if ac is not None else float("inf")


    central_cells, other_cells = [], []
    for cell in wm.shelf_assignments:
        ac = get_access(cell)
        if ac is not None and ac[0] in CENTRAL_AISLE_ROWS:
            central_cells.append(cell)
        else:
            other_cells.append(cell)

    central_cells.sort(key=lambda c: bfs_d(c))
    other_cells.sort(key=lambda c: bfs_d(c))

    gini_before = wm._gini_coefficient(wm.cell_counts)


    all_items: list = []
    for items in wm.shelf_assignments.values():
        all_items.extend(items)

    def by_class(letter):
        return sorted(
            [i for i in all_items if i["ABC_XYZ"][0] == letter],
            key=lambda x: -x.get("Total_Annual_Units", 0.0),
        )

    a_items, b_items, c_items = by_class("A"), by_class("B"), by_class("C")

    cap_map = {cell: len(items) for cell, items in wm.shelf_assignments.items()}
    new_assignment = {cell: [] for cell in wm.shelf_assignments}


    item_pool = list(a_items)
    a_in_central = 0

    def place_at(cell, item):
        item["BFS_Distance"] = bfs_d(cell)
        new_assignment[cell].append(item)


    ptr = 0
    for cell in central_cells:
        if ptr >= len(item_pool):
            break
        for _ in range(cap_map.get(cell, 1)):
            if ptr < len(item_pool):
                place_at(cell, item_pool[ptr])
                ptr += 1
                a_in_central += 1
    leftover_central_cells = [
        c for c in central_cells if len(new_assignment[c]) < cap_map.get(c, 1)
    ]


    a_in_spill = 0
    for cell in other_cells:
        if ptr >= len(item_pool):
            break
        for _ in range(cap_map.get(cell, 1)):
            if ptr < len(item_pool):
                place_at(cell, item_pool[ptr])
                ptr += 1
                a_in_spill += 1

    used_cells = {c for c, lst in new_assignment.items() if lst}
    remaining_other = [c for c in other_cells if c not in used_cells]
    remaining_central = [c for c in central_cells if c not in used_cells]


    b_ptr = 0
    fill_order = remaining_other + remaining_central
    for cell in fill_order:
        if b_ptr >= len(b_items):
            break
        for _ in range(cap_map.get(cell, 1)):
            if b_ptr < len(b_items):
                place_at(cell, b_items[b_ptr])
                b_ptr += 1
    used_cells = {c for c, lst in new_assignment.items() if lst}


    c_ptr = 0
    remaining_for_c = [c for c in (remaining_other + remaining_central) if c not in used_cells]
    for cell in remaining_for_c:
        if c_ptr >= len(c_items):
            break
        for _ in range(cap_map.get(cell, 1)):
            if c_ptr < len(c_items):
                place_at(cell, c_items[c_ptr])
                c_ptr += 1

    wm.shelf_assignments = {k: v for k, v in new_assignment.items() if v}
    wm.build_heatmap(orders)
    gini_after = wm._gini_coefficient(wm.cell_counts)

    return {
        "gini_before": gini_before,
        "gini_after": gini_after,
        "gini_delta": gini_after - gini_before,
        "n_central_shelves": len(central_cells),
        "n_a_in_central": a_in_central,
        "n_a_spilled": a_in_spill,
        "n_a_total": len(a_items),
    }


CAT_COLORS = {
    "AX": "#1a7f37", "AY": "#3fb950", "AZ": "#7ee787",
    "BX": "#e3b341", "BY": "#f0c674", "BZ": "#f7d9a0",
    "CX": "#f85149", "CY": "#ff7b72", "CZ": "#ffa198",
}


def draw_layout(ax, model, title):
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
    ax.set_title(title, fontsize=11, pad=8)


def draw_heatmap_panel(ax, model, title):
    disp = model.cell_counts.astype(float)
    vmax = float(disp.max()) if disp.max() > 0 else 1.0
    mask = np.ma.masked_where(model.grid != 0, disp)
    bg = np.where(model.grid == 1, 0.4, np.where(model.grid == 2, 0.8, 0.05))
    ax.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest", alpha=0.4)
    im = ax.imshow(mask, cmap="YlOrRd",
                   norm=PowerNorm(gamma=0.4, vmin=0, vmax=vmax),
                   alpha=0.9, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, model.grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, model.grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#333", linewidth=0.3)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    gv = model._gini_coefficient(model.cell_counts)
    ax.set_title(f"{title}\nGini={gv:.4f}", fontsize=11, pad=8)
    dr, dc = model.door
    ax.text(dc, dr, "D", ha="center", va="center",
            fontsize=12, color="white", fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.03).set_label(
        f"Visits (max={int(vmax):,})", fontsize=9)


def main():
    wm_base, orders_base = make_wm()
    typ_items = [it for it, _ in Counter(it for o in orders_base for it in o).most_common(ITEMS_PER_ORDER)]
    o0 = orders_base[0]

    p_typ_b = wm_base._route_order_silent(typ_items)
    p_o0_b  = wm_base._route_order_silent(o0)
    base_steps_typ = len(p_typ_b);  base_wpc_typ = compute_wpc(p_typ_b, wm_base.edge_congestion)
    base_steps_o0  = len(p_o0_b);   base_wpc_o0  = compute_wpc(p_o0_b,  wm_base.edge_congestion)
    base_gini      = wm_base._gini_coefficient(wm_base.cell_counts)
    print(f"Baseline | typ: {base_steps_typ}st/{base_wpc_typ:.2f}W | "
          f"o0: {base_steps_o0}st/{base_wpc_o0:.2f}W | Gini={base_gini:.4f}\n")


    wm_cas, _ = make_wm()
    orders_local = wm_cas.generate_orders(num_orders=NUM_ORDERS,
                                          items_per_order=ITEMS_PER_ORDER,
                                          order_seed=ORDER_SEED,
                                          weight_col="Total_Annual_Units", skew=1.0)
    r = central_aisle_slotting(wm_cas, orders_local)
    print(f"CAS partition: {r['n_central_shelves']} central shelves; "
          f"A-items: {r['n_a_in_central']} central + {r['n_a_spilled']} spill = {r['n_a_total']}\n")

    p_typ_cas = wm_cas._route_order_silent(typ_items)
    p_o0_cas  = wm_cas._route_order_silent(o0)
    cas_steps_typ = len(p_typ_cas);  cas_wpc_typ = compute_wpc(p_typ_cas, wm_base.edge_congestion)
    cas_steps_o0  = len(p_o0_cas);   cas_wpc_o0  = compute_wpc(p_o0_cas,  wm_base.edge_congestion)
    cas_gini      = r["gini_after"]

    def pct(d, b): return f"{(d-b)/b*100:+.1f}%"
    print("=== CAS metrics ===")
    print(f"  typ: steps={cas_steps_typ} ({pct(cas_steps_typ, base_steps_typ)})  "
          f"WPC={cas_wpc_typ:.2f} ({pct(cas_wpc_typ, base_wpc_typ)})")
    print(f"  o0:  steps={cas_steps_o0} ({pct(cas_steps_o0, base_steps_o0)})  "
          f"WPC={cas_wpc_o0:.2f} ({pct(cas_wpc_o0, base_wpc_o0)})")
    print(f"  Gini={cas_gini:.4f} ({pct(cas_gini, base_gini)})")


    fig, axes = plt.subplots(1, 2, figsize=(22, 11))
    draw_layout(axes[0], wm_base, "Config A: ABC×XYZ Baseline")
    draw_layout(axes[1], wm_cas,  "Central-Aisle Slotting (CAS)")
    legend_handles = [
        mpatches.Patch(facecolor=CAT_COLORS[c], edgecolor="#444",
                       label=c, linewidth=0.5)
        for c in ["AX", "AY", "AZ", "BX", "BY", "BZ", "CX", "CY", "CZ"]
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=9,
               fontsize=11, framealpha=0.92, bbox_to_anchor=(0.5, -0.04),
               title="SKU Category (A→C = high to low demand, X→Z = low to high variability)",
               title_fontsize=10)
    fig.suptitle("Slotting Layout: Baseline vs Central-Aisle Slotting",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out1 = "final/fig_cas_layout.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out1}")


    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    draw_heatmap_panel(axes[0], wm_base, "Config A: ABC×XYZ Baseline")
    draw_heatmap_panel(axes[1], wm_cas,  "Central-Aisle Slotting (CAS)")
    fig.suptitle("Config A vs Central-Aisle Slotting — Traffic Heatmap",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out2 = "final/fig_cas_heatmap.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")

    with open("cas_results.txt", "w") as f:
        f.write(f"Baseline | typ: {base_steps_typ}st/{base_wpc_typ:.2f}W | "
                f"o0: {base_steps_o0}st/{base_wpc_o0:.2f}W | Gini={base_gini:.4f}\n")
        f.write(f"CAS partition: {r['n_central_shelves']} central shelves; "
                f"A-items: {r['n_a_in_central']} central + {r['n_a_spilled']} spill = {r['n_a_total']}\n")
        f.write(f"CAS      | typ: {cas_steps_typ}st/{cas_wpc_typ:.2f}W "
                f"({pct(cas_steps_typ, base_steps_typ)}/{pct(cas_wpc_typ, base_wpc_typ)}) | "
                f"o0: {cas_steps_o0}st/{cas_wpc_o0:.2f}W "
                f"({pct(cas_steps_o0, base_steps_o0)}/{pct(cas_wpc_o0, base_wpc_o0)}) | "
                f"Gini={cas_gini:.4f}\n")
    print("Saved: cas_results.txt")


if __name__ == "__main__":
    main()
