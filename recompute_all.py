import importlib
from collections import Counter
from typing import Dict

import numpy as np

import warehouse_model
importlib.reload(warehouse_model)
from warehouse_model import WarehouseModel


NUM_ORDERS, ITEMS_PER_ORDER, ORDER_SEED = 10_000, 15, 42
LAM_A = 5.0


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


def pct(d, b):
    return f"{(d - b) / b * 100:+.1f}%"


def report_one(label, wm, baseline_ec, typ_items, o0, base_steps_typ, base_wpc_typ,
               base_steps_o0, base_wpc_o0, baseline_gini, use_ca=False):
    if use_ca:
        p_typ = wm._route_order_congestion_silent(typ_items, lam=LAM_A)
        p_o0  = wm._route_order_congestion_silent(o0,  lam=LAM_A)
    else:
        p_typ = wm._route_order_silent(typ_items)
        p_o0  = wm._route_order_silent(o0)
    metrics = {
        "label": label,
        "steps_typ": len(p_typ),
        "wpc_typ":   compute_wpc(p_typ, baseline_ec),
        "steps_o0":  len(p_o0),
        "wpc_o0":    compute_wpc(p_o0,  baseline_ec),
        "gini":      wm._gini_coefficient(wm.cell_counts),
    }
    print(f"{label:<35} | typ: {metrics['steps_typ']:>3d}st ({pct(metrics['steps_typ'], base_steps_typ):>7}) "
          f"{metrics['wpc_typ']:>6.2f}W ({pct(metrics['wpc_typ'], base_wpc_typ):>7}) | "
          f"o0: {metrics['steps_o0']:>3d}st ({pct(metrics['steps_o0'], base_steps_o0):>7}) "
          f"{metrics['wpc_o0']:>6.2f}W ({pct(metrics['wpc_o0'], base_wpc_o0):>7}) | "
          f"Gini={metrics['gini']:.4f}")
    return metrics


def main():
    wm_base, orders_base = make_wm()
    typ_items = [it for it, _ in Counter(it for o in orders_base for it in o).most_common(ITEMS_PER_ORDER)]
    o0 = orders_base[0]

    p_typ_b = wm_base._route_order_silent(typ_items)
    p_o0_b  = wm_base._route_order_silent(o0)
    base_steps_typ = len(p_typ_b);  base_wpc_typ = compute_wpc(p_typ_b, wm_base.edge_congestion)
    base_steps_o0  = len(p_o0_b);   base_wpc_o0  = compute_wpc(p_o0_b,  wm_base.edge_congestion)
    base_gini      = wm_base._gini_coefficient(wm_base.cell_counts)
    baseline_ec    = wm_base.edge_congestion

    print(f"=== Baseline ABC×XYZ (distance-sorted + BFS) ===")
    print(f"typ: {base_steps_typ}st/{base_wpc_typ:.2f}W | "
          f"o0: {base_steps_o0}st/{base_wpc_o0:.2f}W | Gini={base_gini:.4f}\n")


    print("=== Config A (Dijkstra-NN, λ=5.0) on baseline slotting ===")
    print(f"{'Variant':<35} | typ: ...                          | o0: ...                          | Gini")
    print("-" * 160)
    cfg_a = report_one("Config A (CA λ=5, Dijkstra-NN)", wm_base, baseline_ec, typ_items, o0,
                       base_steps_typ, base_wpc_typ, base_steps_o0, base_wpc_o0, base_gini, use_ca=True)


    print("\n=== Config A batch (K=10000) ===")
    bfs_lengths = wm_base.batch_route_lengths(orders_base, lam=0.0)
    ca_lengths  = wm_base.batch_route_lengths(orders_base, lam=LAM_A)
    bfs_mean = float(np.mean(bfs_lengths)); bfs_L95 = int(np.percentile(bfs_lengths, 95))
    ca_mean  = float(np.mean(ca_lengths));  ca_L95  = int(np.percentile(ca_lengths, 95))

    bfs_wpc_list = [compute_wpc(wm_base._route_order_silent(o), baseline_ec) for o in orders_base]
    ca_wpc_list  = [compute_wpc(wm_base._route_order_congestion_silent(o, lam=LAM_A), baseline_ec)
                    for o in orders_base]
    bfs_wpc_mean = float(np.mean(bfs_wpc_list)); ca_wpc_mean = float(np.mean(ca_wpc_list))

    print(f"Baseline B (BFS): mean steps={bfs_mean:.1f}, L95={bfs_L95}, mean WPC={bfs_wpc_mean:.3f}")
    print(f"Config A (CA):    mean steps={ca_mean:.1f}, L95={ca_L95}, mean WPC={ca_wpc_mean:.3f}")
    print(f"Δ (CA - Baseline): mean={ca_mean - bfs_mean:+.1f} ({pct(ca_mean, bfs_mean)}), "
          f"L95={ca_L95 - bfs_L95:+d} ({pct(ca_L95, bfs_L95)}), "
          f"WPC={ca_wpc_mean - bfs_wpc_mean:+.3f} ({pct(ca_wpc_mean, bfs_wpc_mean)})")


    print("\n=== Five Slotting Strategies (BFS routing on new slotting) ===")
    print(f"{'Variant':<35} | typ: ...                          | o0: ...                          | Gini")
    print("-" * 160)

    results = {"Baseline": {"label": "Baseline ABC×XYZ", "steps_typ": base_steps_typ,
                            "wpc_typ": base_wpc_typ, "steps_o0": base_steps_o0,
                            "wpc_o0": base_wpc_o0, "gini": base_gini}}

    def fresh_wm():
        wm = WarehouseModel(rows=52, cols=52, shelf_length=4, shelf_gap=1,
                            shelf_rows_height=2, aisle_width=2, margin=1,
                            shelf_capacity=1, seed=6, door=(23, 0))
        wm.generate(); wm.build_graph()
        wm.abc_classify(csv_path="abc_xyz_dataset.csv",
                        a_pct=0.20, b_pct=0.30, x_cv=0.10, y_cv=0.25)
        wm.assign_items_to_shelves()
        os = wm.generate_orders(num_orders=NUM_ORDERS,
                                items_per_order=ITEMS_PER_ORDER,
                                order_seed=ORDER_SEED,
                                weight_col="Total_Annual_Units", skew=1.0)
        wm.build_heatmap(os)
        return wm, os


    wm_dwc, o_dwc = fresh_wm()
    wm_dwc.dwc_slotting(o_dwc, dist_tol=5.0, max_swaps=None, rebuild=True, verbose=False)
    results["DWC"] = report_one("DWC (Δ=5)", wm_dwc, baseline_ec, typ_items, o0,
                                 base_steps_typ, base_wpc_typ, base_steps_o0, base_wpc_o0, base_gini)


    wm_hzr, o_hzr = fresh_wm()
    wm_hzr.hot_zone_relegation(o_hzr, hot_pct=0.15, rebuild=True, verbose=False)
    results["HZR"] = report_one("HZR (h=0.15)", wm_hzr, baseline_ec, typ_items, o0,
                                 base_steps_typ, base_wpc_typ, base_steps_o0, base_wpc_o0, base_gini)


    wm_taps, o_taps = fresh_wm()
    wm_taps.taps_slotting(o_taps, alpha=3.0, rebuild=True, verbose=False)
    results["TAPS"] = report_one("TAPS (α=3)", wm_taps, baseline_ec, typ_items, o0,
                                  base_steps_typ, base_wpc_typ, base_steps_o0, base_wpc_o0, base_gini)


    wm_ab, o_ab = fresh_wm()
    wm_ab.aisle_balanced_slotting(o_ab, rebuild=True, verbose=False)
    results["AB"] = report_one("Aisle-Balanced (f_A=0.25)", wm_ab, baseline_ec, typ_items, o0,
                                base_steps_typ, base_wpc_typ, base_steps_o0, base_wpc_o0, base_gini)


    wm_aahms, o_aahms = fresh_wm()
    wm_aahms.affinity_aware_slotting(o_aahms, top_k_pairs=150, max_swaps=400,
                                      congestion_tol=0.10, rebuild=True, verbose=False)
    results["AAHMS"] = report_one("AAHMS (K=150)", wm_aahms, baseline_ec, typ_items, o0,
                                   base_steps_typ, base_wpc_typ, base_steps_o0, base_wpc_o0, base_gini)


    print("\n=== CAS (Central-Aisle Slotting) ===")
    from central_aisle_slotting import central_aisle_slotting as cas_fn
    wm_cas, o_cas = fresh_wm()
    cas_fn(wm_cas, o_cas)
    results["CAS"] = report_one("CAS (central-aisle)", wm_cas, baseline_ec, typ_items, o0,
                                 base_steps_typ, base_wpc_typ, base_steps_o0, base_wpc_o0, base_gini)


    with open("recomputed_metrics.txt", "w") as f:
        f.write(f"=== Baseline ABC×XYZ (distance-sorted + BFS) ===\n")
        f.write(f"typ: {base_steps_typ}st/{base_wpc_typ:.2f}W | "
                f"o0: {base_steps_o0}st/{base_wpc_o0:.2f}W | Gini={base_gini:.4f}\n\n")
        f.write(f"=== Config A batch (K=10000) ===\n")
        f.write(f"Baseline B (BFS): mean steps={bfs_mean:.1f}, L95={bfs_L95}, mean WPC={bfs_wpc_mean:.3f}\n")
        f.write(f"Config A (CA):    mean steps={ca_mean:.1f}, L95={ca_L95}, mean WPC={ca_wpc_mean:.3f}\n")
        f.write(f"Δ (CA - Baseline): mean={ca_mean - bfs_mean:+.1f} ({pct(ca_mean, bfs_mean)}), "
                f"L95={ca_L95 - bfs_L95:+d} ({pct(ca_L95, bfs_L95)}), "
                f"WPC={ca_wpc_mean - bfs_wpc_mean:+.3f} ({pct(ca_wpc_mean, bfs_wpc_mean)})\n\n")
        f.write("=== Per-strategy metrics (BFS routing on new slotting) ===\n")
        f.write(f"{'Variant':<28} {'typ_st':>6} {'typ_WPC':>8} {'o0_st':>6} {'o0_WPC':>8} {'Gini':>8}\n")
        for k, v in results.items():
            f.write(f"{v['label']:<28} {v['steps_typ']:>6d} {v['wpc_typ']:>8.2f} "
                    f"{v['steps_o0']:>6d} {v['wpc_o0']:>8.2f} {v['gini']:>8.4f}\n")
        f.write(f"\nConfig A (CA on baseline slotting): typ_st={cfg_a['steps_typ']}, "
                f"typ_WPC={cfg_a['wpc_typ']:.2f}, o0_st={cfg_a['steps_o0']}, "
                f"o0_WPC={cfg_a['wpc_o0']:.2f}, Gini={cfg_a['gini']:.4f}\n")
    print("\nSaved: recomputed_metrics.txt")


if __name__ == "__main__":
    main()
