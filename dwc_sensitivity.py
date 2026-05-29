import matplotlib
matplotlib.use("Agg")
import importlib
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

import warehouse_model
importlib.reload(warehouse_model)
from warehouse_model import WarehouseModel


NUM_ORDERS, ITEMS_PER_ORDER, ORDER_SEED = 10_000, 15, 42


def make_wm():
    wm = WarehouseModel(rows=52, cols=52, shelf_length=4, shelf_gap=1,
                        shelf_rows_height=2, aisle_width=2, margin=1,
                        shelf_capacity=1, seed=6, door=(23, 0))
    wm.generate(); wm.build_graph()
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


def main():
    wm_base, orders_base = make_wm()
    typ_items = [it for it, _ in Counter(it for o in orders_base for it in o).most_common(ITEMS_PER_ORDER)]
    p_typ_b = wm_base._route_order_silent(typ_items)
    base_steps = len(p_typ_b); base_wpc = compute_wpc(p_typ_b, wm_base.edge_congestion)
    base_gini = wm_base._gini_coefficient(wm_base.cell_counts)
    print(f"Baseline | steps={base_steps}, WPC={base_wpc:.2f}, Gini={base_gini:.4f}\n")

    sweep = [5, 10, 15, 20, 30, 50]
    rows = []
    for delta in sweep:
        wm, _ = make_wm()
        orders = wm.generate_orders(num_orders=NUM_ORDERS,
                                     items_per_order=ITEMS_PER_ORDER,
                                     order_seed=ORDER_SEED,
                                     weight_col="Total_Annual_Units", skew=1.0)
        r = wm.dwc_slotting(orders, dist_tol=float(delta), max_swaps=None,
                             rebuild=True, verbose=False)
        p_typ = wm._route_order_silent(typ_items)
        rows.append({
            "delta": delta, "swaps": r["n_swaps"],
            "steps": len(p_typ), "wpc": compute_wpc(p_typ, wm_base.edge_congestion),
            "gini": r["gini_after"],
        })
        rows[-1]["d_wpc_pct"]   = (rows[-1]["wpc"]   - base_wpc)   / base_wpc   * 100
        rows[-1]["d_steps_pct"] = (rows[-1]["steps"] - base_steps) / base_steps * 100
        print(f"Δ={delta:>2}: swaps={r['n_swaps']:>3d} | steps={len(p_typ):>4d} ({rows[-1]['d_steps_pct']:+.0f}%) "
              f"| WPC={rows[-1]['wpc']:.2f} ({rows[-1]['d_wpc_pct']:+.0f}%) | Gini={r['gini_after']:.4f}")


    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    xs = [r["delta"] for r in sweep_idx(rows)]
    ax = axes[0]
    ax.plot(xs, [r["wpc"] for r in rows], "o-", color="#c0392b", lw=2, ms=7)
    ax.axhline(base_wpc, color="#888", ls="--", lw=1.2, label=f"Baseline = {base_wpc:.1f}")
    for x, y in zip(xs, [r["wpc"] for r in rows]):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel(r"$\Delta$ (BFS steps)"); ax.set_ylabel("WPC (typical order)")
    ax.set_title("Weighted Path Congestion"); ax.legend(fontsize=8); ax.grid(alpha=0.4)
    ax.axvspan(0, 10, alpha=0.18, color="gray")

    ax = axes[1]
    ax.plot(xs, [r["steps"] for r in rows], "s-", color="#2980b9", lw=2, ms=7)
    ax.axhline(base_steps, color="#888", ls="--", lw=1.2, label=f"Baseline = {base_steps}")
    for x, y in zip(xs, [r["steps"] for r in rows]):
        ax.annotate(f"{y}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel(r"$\Delta$ (BFS steps)"); ax.set_ylabel("Route length (steps)")
    ax.set_title("Route Length"); ax.legend(fontsize=8); ax.grid(alpha=0.4)
    ax.axvspan(0, 10, alpha=0.18, color="gray")

    ax = axes[2]
    ax.plot(xs, [r["gini"] for r in rows], "^-", color="#27ae60", lw=2, ms=7)
    ax.axhline(base_gini, color="#888", ls="--", lw=1.2, label=f"Baseline = {base_gini:.3f}")
    for x, y in zip(xs, [r["gini"] for r in rows]):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel(r"$\Delta$ (BFS steps)"); ax.set_ylabel("Gini coefficient")
    ax.set_title("Traffic Gini"); ax.legend(fontsize=8); ax.grid(alpha=0.4)
    ax.axvspan(0, 10, alpha=0.18, color="gray")

    fig.suptitle("DWC Slotting sensitivity to Delta (K=10000, typical order)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = "final/fig_dwc_delta_sensitivity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nSaved: {out}")

    with open("dwc_sensitivity_results.txt", "w") as f:
        f.write(f"Baseline | steps={base_steps}, WPC={base_wpc:.2f}, Gini={base_gini:.4f}\n\n")
        f.write(f"{'Δ':>4} {'swaps':>6} {'steps':>6} {'d_st%':>7} {'WPC':>8} {'d_W%':>7} {'Gini':>8}\n")
        for r in rows:
            f.write(f"{r['delta']:>4d} {r['swaps']:>6d} {r['steps']:>6d} {r['d_steps_pct']:>+6.1f}% "
                    f"{r['wpc']:>8.2f} {r['d_wpc_pct']:>+6.1f}% {r['gini']:>8.4f}\n")
    print("Saved: dwc_sensitivity_results.txt")


def sweep_idx(rows):
    return rows


if __name__ == "__main__":
    main()
