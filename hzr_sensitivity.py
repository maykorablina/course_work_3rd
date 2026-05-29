import matplotlib
matplotlib.use("Agg")

import importlib
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import warehouse_model
importlib.reload(warehouse_model)
from warehouse_model import WarehouseModel


NUM_ORDERS, ITEMS_PER_ORDER, ORDER_SEED = 10_000, 15, 42


def build_baseline():
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


def run_hzr(hot_pct, baseline_ec, typical_items):
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
    r = wm.hot_zone_relegation(orders, hot_pct=hot_pct, rebuild=True, verbose=False)
    path_typ = wm._route_order_silent(typical_items)
    wpc_typ = compute_wpc(path_typ, baseline_ec)
    return {
        "hot_pct": hot_pct,
        "n_hot": r["n_hot_cells"],
        "gini_before": r["gini_before"],
        "gini_after": r["gini_after"],
        "wpc_typ": wpc_typ,
        "steps_typ": len(path_typ),
        "c_in_hot": r["c_in_hot"],
        "a_in_cool": r["a_in_cool"],
    }


def main():
    wm_base, orders_base = build_baseline()
    typ_items = [it for it, _ in Counter(it for o in orders_base for it in o).most_common(ITEMS_PER_ORDER)]
    path_typ_base = wm_base._route_order_silent(typ_items)
    base_steps = len(path_typ_base)
    base_wpc = compute_wpc(path_typ_base, wm_base.edge_congestion)
    base_gini = wm_base._gini_coefficient(wm_base.cell_counts)
    print(f"Baseline | steps={base_steps}, WPC={base_wpc:.2f}, Gini={base_gini:.4f}")

    sweep = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    rows = []
    for hp in sweep:
        r = run_hzr(hp, wm_base.edge_congestion, typ_items)
        r["d_wpc_pct"] = (r["wpc_typ"] - base_wpc) / base_wpc * 100
        r["d_steps_pct"] = (r["steps_typ"] - base_steps) / base_steps * 100
        rows.append(r)
        print(f"hot_pct={hp:.2f} | n_hot={r['n_hot']:3d} "
              f"Gini={r['gini_after']:.4f} | steps={r['steps_typ']:3d} "
              f"({r['d_steps_pct']:+.0f}%) | WPC={r['wpc_typ']:.2f} ({r['d_wpc_pct']:+.0f}%)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    xs = [r["hot_pct"] * 100 for r in rows]
    wpcs = [r["wpc_typ"] for r in rows]
    steps = [r["steps_typ"] for r in rows]
    ginis = [r["gini_after"] for r in rows]

    ax = axes[0]
    ax.plot(xs, wpcs, "o-", color="#c0392b", lw=2, ms=8)
    ax.axhline(base_wpc, color="#888", ls="--", lw=1.2, label=f"Baseline = {base_wpc:.1f}")
    for x, y in zip(xs, wpcs):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel(r"hot\_pct (%)")
    ax.set_ylabel("WPC (typical order)")
    ax.set_title("Weighted Path Congestion")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.4)

    ax = axes[1]
    ax.plot(xs, steps, "s-", color="#2980b9", lw=2, ms=8)
    ax.axhline(base_steps, color="#888", ls="--", lw=1.2, label=f"Baseline = {base_steps}")
    for x, y in zip(xs, steps):
        ax.annotate(f"{y}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel(r"hot\_pct (%)")
    ax.set_ylabel("Route length (steps)")
    ax.set_title("Route Length")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.4)

    ax = axes[2]
    ax.plot(xs, ginis, "^-", color="#27ae60", lw=2, ms=8)
    ax.axhline(base_gini, color="#888", ls="--", lw=1.2, label=f"Baseline = {base_gini:.3f}")
    for x, y in zip(xs, ginis):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel(r"hot\_pct (%)")
    ax.set_ylabel("Gini coefficient")
    ax.set_title("Traffic Gini (lower = more uniform)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.4)

    fig.suptitle("Hot-Zone Relegation sensitivity to hot_pct (K=10000, typical order)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = "final/fig_hzr_sensitivity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    with open("hzr_sensitivity_results.txt", "w") as f:
        f.write(f"Baseline | steps={base_steps}, WPC={base_wpc:.2f}, Gini={base_gini:.4f}\n\n")
        f.write(f"{'hot_pct':>8} {'n_hot':>6} {'Gini':>8} {'steps':>6} {'d_steps%':>9} {'WPC':>8} {'d_wpc%':>9} {'C_in_hot':>9} {'A_in_cool':>10}\n")
        for r in rows:
            f.write(f"{r['hot_pct']:>8.2f} {r['n_hot']:>6d} {r['gini_after']:>8.4f} "
                    f"{r['steps_typ']:>6d} {r['d_steps_pct']:>+8.1f}% {r['wpc_typ']:>8.2f} "
                    f"{r['d_wpc_pct']:>+8.1f}% {r['c_in_hot']:>9d} {r['a_in_cool']:>10d}\n")
    print("Saved: hzr_sensitivity_results.txt")


if __name__ == "__main__":
    main()
