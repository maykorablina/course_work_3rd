import matplotlib
matplotlib.use("Agg")

import importlib
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

import warehouse_model
importlib.reload(warehouse_model)
from warehouse_model import WarehouseModel


NUM_ORDERS, ITEMS_PER_ORDER, ORDER_SEED = 10_000, 15, 42
A_FRAC_DEFAULT = 0.25
B_FRAC = 0.30


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


def aisle_balanced_parametric(wm: WarehouseModel, orders, a_frac: float, b_frac: float = B_FRAC):

    if wm.heatmap is None:
        wm.build_heatmap(orders)

    gini_before = wm._gini_coefficient(wm.cell_counts)
    dist_corr = wm.corridor_distances()
    grid = wm.grid
    rows_, cols_ = grid.shape

    def bfs_d(cell):
        r, c = cell
        best = float("inf")
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < rows_ and 0 <= cc < cols_ and grid[rr, cc] in (0, 2):
                best = min(best, dist_corr[rr, cc])
        return best

    aisle_shelves: Dict[int, list] = defaultdict(list)
    for cell in wm.shelf_assignments:
        aisle_shelves[cell[0]].append(cell)

    aisle_dist = {}
    for aisle_row, cells in aisle_shelves.items():
        dists = [bfs_d(c) for c in cells]
        finite = [d for d in dists if d < float("inf")]
        aisle_dist[aisle_row] = sum(finite) / len(finite) if finite else float("inf")

    sorted_aisles = sorted(aisle_shelves.keys(), key=lambda a: aisle_dist[a])
    n_aisles = len(sorted_aisles)

    n_a = max(3, int(n_aisles * a_frac))
    n_b = int(n_aisles * b_frac)
    a_aisles = sorted_aisles[:n_a]
    b_aisles = sorted_aisles[n_a:n_a + n_b]
    c_aisles = sorted_aisles[n_a + n_b:]

    def shelves_for_aisles(aisle_rows):
        slots = []
        shelves_by_aisle = [
            sorted(aisle_shelves[a], key=lambda c: bfs_d(c))
            for a in aisle_rows
        ]
        max_per = max((len(s) for s in shelves_by_aisle), default=0)
        for col_idx in range(max_per):
            for aisle_list in shelves_by_aisle:
                if col_idx < len(aisle_list):
                    slots.append(aisle_list[col_idx])
        return slots

    a_slots = shelves_for_aisles(a_aisles)
    b_slots = shelves_for_aisles(b_aisles)
    c_slots = shelves_for_aisles(c_aisles)

    all_items: list = []
    for items in wm.shelf_assignments.values():
        all_items.extend(items)

    def by_class(letter):
        return sorted([i for i in all_items if i["ABC_XYZ"][0] == letter],
                      key=lambda x: -x.get("Total_Annual_Units", 0.0))

    a_items, b_items, c_items = by_class("A"), by_class("B"), by_class("C")
    cap_map = {cell: len(items) for cell, items in wm.shelf_assignments.items()}
    new_assignment = {cell: [] for cell in wm.shelf_assignments}

    def fill(slots, primary, overflow):
        pool = primary + overflow
        ptr = 0
        for cell in slots:
            for _ in range(cap_map.get(cell, 1)):
                if ptr < len(pool):
                    item = pool[ptr]
                    item["BFS_Distance"] = bfs_d(cell)
                    new_assignment[cell].append(item)
                    ptr += 1
        return pool[ptr:]

    leftover_a = fill(a_slots, a_items, [])
    leftover_b = fill(b_slots, b_items, leftover_a)
    fill(c_slots, c_items, leftover_b)

    wm.shelf_assignments = {k: v for k, v in new_assignment.items() if v}
    wm.build_heatmap(orders)
    gini_after = wm._gini_coefficient(wm.cell_counts)
    return {
        "gini_before": gini_before, "gini_after": gini_after,
        "n_a_aisles": len(a_aisles), "n_b_aisles": len(b_aisles), "n_c_aisles": len(c_aisles),
    }


def sweep_ab(typ_items, o0, baseline_ec, base_steps_typ, base_wpc_typ,
             base_steps_o0, base_wpc_o0, base_gini):
    sweep = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    rows = []
    for f in sweep:
        wm, _ = make_wm()
        orders_local = wm.generate_orders(num_orders=NUM_ORDERS,
                                          items_per_order=ITEMS_PER_ORDER,
                                          order_seed=ORDER_SEED,
                                          weight_col="Total_Annual_Units", skew=1.0)
        r = aisle_balanced_parametric(wm, orders_local, a_frac=f)
        p_typ = wm._route_order_silent(typ_items)
        p_o0 = wm._route_order_silent(o0)
        rows.append({
            "frac": f,
            "n_a": r["n_a_aisles"], "n_b": r["n_b_aisles"], "n_c": r["n_c_aisles"],
            "gini": r["gini_after"],
            "steps_typ": len(p_typ), "wpc_typ": compute_wpc(p_typ, baseline_ec),
            "steps_o0": len(p_o0), "wpc_o0": compute_wpc(p_o0, baseline_ec),
        })
        rows[-1]["d_wpc_typ"]   = (rows[-1]["wpc_typ"]   - base_wpc_typ)   / base_wpc_typ   * 100
        rows[-1]["d_steps_typ"] = (rows[-1]["steps_typ"] - base_steps_typ) / base_steps_typ * 100
        rows[-1]["d_wpc_o0"]    = (rows[-1]["wpc_o0"]    - base_wpc_o0)    / base_wpc_o0    * 100
        rows[-1]["d_steps_o0"]  = (rows[-1]["steps_o0"]  - base_steps_o0)  / base_steps_o0  * 100
        print(f"AB a_frac={f:.2f} | n_a={r['n_a_aisles']:2d} "
              f"| Gini={r['gini_after']:.4f} "
              f"| typ: {len(p_typ):3d}st/{rows[-1]['wpc_typ']:.2f}W "
              f"| o0: {len(p_o0):3d}st/{rows[-1]['wpc_o0']:.2f}W")


    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    xs = [r["frac"] * 100 for r in rows]
    ax = axes[0]
    ax.plot(xs, [r["wpc_typ"] for r in rows], "o-", color="#c0392b", lw=2, ms=7, label="Typical")
    ax.plot(xs, [r["wpc_o0"]  for r in rows], "s--", color="#8e44ad", lw=1.6, ms=6, label="$o_0$")
    ax.axhline(base_wpc_typ, color="#c0392b", ls=":", lw=1, alpha=0.6, label=f"Baseline typ = {base_wpc_typ:.1f}")
    ax.axhline(base_wpc_o0,  color="#8e44ad", ls=":", lw=1, alpha=0.6, label=f"Baseline $o_0$ = {base_wpc_o0:.1f}")
    ax.axvline(25, color="#27ae60", ls="--", lw=1.2, alpha=0.7)
    ax.set_xlabel(r"A-aisle fraction (%)")
    ax.set_ylabel("WPC")
    ax.set_title("Weighted Path Congestion")
    ax.legend(fontsize=8); ax.grid(alpha=0.4)

    ax = axes[1]
    ax.plot(xs, [r["steps_typ"] for r in rows], "o-", color="#2980b9", lw=2, ms=7, label="Typical")
    ax.plot(xs, [r["steps_o0"]  for r in rows], "s--", color="#16a085", lw=1.6, ms=6, label="$o_0$")
    ax.axhline(base_steps_typ, color="#2980b9", ls=":", lw=1, alpha=0.6, label=f"Baseline typ = {base_steps_typ}")
    ax.axhline(base_steps_o0,  color="#16a085", ls=":", lw=1, alpha=0.6, label=f"Baseline $o_0$ = {base_steps_o0}")
    ax.axvline(25, color="#27ae60", ls="--", lw=1.2, alpha=0.7)
    ax.set_xlabel(r"A-aisle fraction (%)")
    ax.set_ylabel("Route length (steps)")
    ax.set_title("Route Length")
    ax.legend(fontsize=8); ax.grid(alpha=0.4)

    ax = axes[2]
    ax.plot(xs, [r["gini"] for r in rows], "^-", color="#27ae60", lw=2, ms=7)
    ax.axhline(base_gini, color="#888", ls="--", lw=1.2, label=f"Baseline = {base_gini:.3f}")
    ax.axvline(25, color="#27ae60", ls="--", lw=1.2, alpha=0.5)
    for x, y in zip(xs, [r["gini"] for r in rows]):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7)
    ax.set_xlabel(r"A-aisle fraction (%)")
    ax.set_ylabel("Gini")
    ax.set_title("Traffic Gini")
    ax.legend(fontsize=8); ax.grid(alpha=0.4)

    fig.suptitle("Aisle-Balanced Slotting sensitivity to A-aisle fraction (K=10000)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = "final/fig_ab_sensitivity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")
    return rows


def sweep_aahms(typ_items, o0, baseline_ec, base_steps_typ, base_wpc_typ,
                base_steps_o0, base_wpc_o0, base_gini):
    sweep = [25, 50, 100, 150, 250, 400]
    rows = []
    for k in sweep:
        wm, _ = make_wm()
        orders_local = wm.generate_orders(num_orders=NUM_ORDERS,
                                          items_per_order=ITEMS_PER_ORDER,
                                          order_seed=ORDER_SEED,
                                          weight_col="Total_Annual_Units", skew=1.0)
        r = wm.affinity_aware_slotting(orders_local, top_k_pairs=k, max_swaps=400,
                                        congestion_tol=0.10, rebuild=True, verbose=False)
        p_typ = wm._route_order_silent(typ_items)
        p_o0 = wm._route_order_silent(o0)
        rows.append({
            "k": k, "n_swaps": r["n_swaps"], "gini": r["gini_after"],
            "steps_typ": len(p_typ), "wpc_typ": compute_wpc(p_typ, baseline_ec),
            "steps_o0": len(p_o0), "wpc_o0": compute_wpc(p_o0, baseline_ec),
        })
        rows[-1]["d_wpc_typ"]   = (rows[-1]["wpc_typ"]   - base_wpc_typ)   / base_wpc_typ   * 100
        rows[-1]["d_steps_typ"] = (rows[-1]["steps_typ"] - base_steps_typ) / base_steps_typ * 100
        rows[-1]["d_wpc_o0"]    = (rows[-1]["wpc_o0"]    - base_wpc_o0)    / base_wpc_o0    * 100
        rows[-1]["d_steps_o0"]  = (rows[-1]["steps_o0"]  - base_steps_o0)  / base_steps_o0  * 100
        print(f"AAHMS K={k:>3d} | swaps={r['n_swaps']:>3d} | Gini={r['gini_after']:.4f} "
              f"| typ: {len(p_typ):3d}st/{rows[-1]['wpc_typ']:.2f}W "
              f"| o0: {len(p_o0):3d}st/{rows[-1]['wpc_o0']:.2f}W")


    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    xs = [r["k"] for r in rows]
    ax = axes[0]
    ax.plot(xs, [r["wpc_typ"] for r in rows], "o-", color="#c0392b", lw=2, ms=7, label="Typical")
    ax.plot(xs, [r["wpc_o0"]  for r in rows], "s--", color="#8e44ad", lw=1.6, ms=6, label="$o_0$")
    ax.axhline(base_wpc_typ, color="#c0392b", ls=":", lw=1, alpha=0.6, label=f"Baseline typ = {base_wpc_typ:.1f}")
    ax.axhline(base_wpc_o0,  color="#8e44ad", ls=":", lw=1, alpha=0.6, label=f"Baseline $o_0$ = {base_wpc_o0:.1f}")
    ax.axvline(150, color="#27ae60", ls="--", lw=1.2, alpha=0.7)
    ax.set_xlabel(r"top-$K$ pairs")
    ax.set_ylabel("WPC")
    ax.set_title("Weighted Path Congestion")
    ax.legend(fontsize=8); ax.grid(alpha=0.4)

    ax = axes[1]
    ax.plot(xs, [r["steps_typ"] for r in rows], "o-", color="#2980b9", lw=2, ms=7, label="Typical")
    ax.plot(xs, [r["steps_o0"]  for r in rows], "s--", color="#16a085", lw=1.6, ms=6, label="$o_0$")
    ax.axhline(base_steps_typ, color="#2980b9", ls=":", lw=1, alpha=0.6, label=f"Baseline typ = {base_steps_typ}")
    ax.axhline(base_steps_o0,  color="#16a085", ls=":", lw=1, alpha=0.6, label=f"Baseline $o_0$ = {base_steps_o0}")
    ax.axvline(150, color="#27ae60", ls="--", lw=1.2, alpha=0.7)
    ax.set_xlabel(r"top-$K$ pairs")
    ax.set_ylabel("Route length (steps)")
    ax.set_title("Route Length")
    ax.legend(fontsize=8); ax.grid(alpha=0.4)

    ax = axes[2]
    ax.plot(xs, [r["n_swaps"] for r in rows], "d-", color="#d68910", lw=2, ms=7)
    ax.axvline(150, color="#27ae60", ls="--", lw=1.2, alpha=0.5)
    for x, y in zip(xs, [r["n_swaps"] for r in rows]):
        ax.annotate(f"{y}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel(r"top-$K$ pairs")
    ax.set_ylabel("Accepted swaps")
    ax.set_title("Number of accepted swaps")
    ax.grid(alpha=0.4)

    fig.suptitle("AAHMS sensitivity to top-K co-occurring pairs (K=10000)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = "final/fig_aahms_sensitivity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")
    return rows


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

    print("=== AISLE-BALANCED SWEEP ===")
    ab_rows = sweep_ab(typ_items, o0, wm_base.edge_congestion,
                       base_steps_typ, base_wpc_typ, base_steps_o0, base_wpc_o0, base_gini)

    print("\n=== AAHMS SWEEP ===")
    aahms_rows = sweep_aahms(typ_items, o0, wm_base.edge_congestion,
                             base_steps_typ, base_wpc_typ, base_steps_o0, base_wpc_o0, base_gini)

    with open("ab_aahms_results.txt", "w") as f:
        f.write(f"Baseline | typ: {base_steps_typ}st/{base_wpc_typ:.2f}W | "
                f"o0: {base_steps_o0}st/{base_wpc_o0:.2f}W | Gini={base_gini:.4f}\n\n")
        f.write("=== AISLE-BALANCED (sweep over A-aisle fraction) ===\n")
        f.write(f"{'frac':>5} {'nA':>3} {'nB':>3} {'nC':>3} {'Gini':>7} | "
                f"{'tp_st':>5} {'d%':>6} {'tp_W':>6} {'d%':>6} | "
                f"{'o0_st':>5} {'d%':>6} {'o0_W':>6} {'d%':>6}\n")
        for r in ab_rows:
            f.write(f"{r['frac']:>5.2f} {r['n_a']:>3d} {r['n_b']:>3d} {r['n_c']:>3d} "
                    f"{r['gini']:>7.4f} | "
                    f"{r['steps_typ']:>5d} {r['d_steps_typ']:>+5.1f}% {r['wpc_typ']:>6.2f} {r['d_wpc_typ']:>+5.1f}% | "
                    f"{r['steps_o0']:>5d} {r['d_steps_o0']:>+5.1f}% {r['wpc_o0']:>6.2f} {r['d_wpc_o0']:>+5.1f}%\n")
        f.write("\n=== AAHMS (sweep over top-K pairs) ===\n")
        f.write(f"{'K':>4} {'swaps':>5} {'Gini':>7} | "
                f"{'tp_st':>5} {'d%':>6} {'tp_W':>6} {'d%':>6} | "
                f"{'o0_st':>5} {'d%':>6} {'o0_W':>6} {'d%':>6}\n")
        for r in aahms_rows:
            f.write(f"{r['k']:>4d} {r['n_swaps']:>5d} {r['gini']:>7.4f} | "
                    f"{r['steps_typ']:>5d} {r['d_steps_typ']:>+5.1f}% {r['wpc_typ']:>6.2f} {r['d_wpc_typ']:>+5.1f}% | "
                    f"{r['steps_o0']:>5d} {r['d_steps_o0']:>+5.1f}% {r['wpc_o0']:>6.2f} {r['d_wpc_o0']:>+5.1f}%\n")
    print("Saved: ab_aahms_results.txt")


if __name__ == "__main__":
    main()
