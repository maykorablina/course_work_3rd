import importlib
from collections import Counter
import numpy as np

import warehouse_model
importlib.reload(warehouse_model)
from warehouse_model import WarehouseModel
from central_aisle_slotting import central_aisle_slotting


NUM_ORDERS, ITEMS_PER_ORDER, ORDER_SEED, LAM = 10_000, 15, 42, 5.0


def fresh_wm():
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


def evaluate(label, wm, orders_local, typ_items, o0, ec_base, lam):
    if lam == 0.0:
        route = wm._route_order_silent
    else:
        route = lambda o: wm._route_order_congestion_silent(o, lam=lam)

    p_typ = route(typ_items)
    p_o0  = route(o0)

    lengths = []
    wpcs    = []
    for o in orders_local:
        p = route(o)
        lengths.append(len(p))
        wpcs.append(compute_wpc(p, ec_base))

    return {
        "label": label,
        "steps_typ": len(p_typ), "wpc_typ": compute_wpc(p_typ, ec_base),
        "steps_o0":  len(p_o0),  "wpc_o0":  compute_wpc(p_o0,  ec_base),
        "mean_steps": float(np.mean(lengths)),
        "L95":        int(np.percentile(lengths, 95)),
        "mean_wpc":   float(np.mean(wpcs)),
        "gini":       wm._gini_coefficient(wm.cell_counts),
    }


def pct(d, b): return f"{(d - b) / b * 100:+.1f}%"


def main():

    print("Building Baseline...", flush=True)
    wm_base, orders_base = fresh_wm()
    ec_base = wm_base.edge_congestion
    typ_items = [it for it, _ in
                 Counter(it for o in orders_base for it in o).most_common(ITEMS_PER_ORDER)]
    o0 = orders_base[0]


    def build_baseline():
        wm, o = fresh_wm()
        return wm, o, "Baseline"

    def build_dwc():
        wm, o = fresh_wm()
        wm.dwc_slotting(o, dist_tol=5.0, max_swaps=None, rebuild=True, verbose=False)
        return wm, o, "DWC (Δ=5)"

    def build_hzr():
        wm, o = fresh_wm()
        wm.hot_zone_relegation(o, hot_pct=0.15, rebuild=True, verbose=False)
        return wm, o, "HZR (h=0.15)"

    def build_taps():
        wm, o = fresh_wm()
        wm.taps_slotting(o, alpha=3.0, rebuild=True, verbose=False)
        return wm, o, "TAPS (α=3)"

    def build_ab():
        wm, o = fresh_wm()
        wm.aisle_balanced_slotting(o, rebuild=True, verbose=False)
        return wm, o, "Aisle-Balanced"

    def build_aahms():
        wm, o = fresh_wm()
        wm.affinity_aware_slotting(o, top_k_pairs=150, max_swaps=400,
                                    congestion_tol=0.10, rebuild=True, verbose=False)
        return wm, o, "AAHMS (K=150)"

    def build_cas():
        wm, o = fresh_wm()
        central_aisle_slotting(wm, o)
        return wm, o, "CAS"

    builders = [build_baseline, build_dwc, build_hzr, build_taps,
                build_ab, build_aahms, build_cas]


    print("\n=== 7 slottings × 2 routings ===\n", flush=True)
    results = []
    for builder in builders:
        wm, orders_local, name = builder()
        r_bfs = evaluate(f"{name}",          wm, orders_local, typ_items, o0, ec_base, lam=0.0)

        wm2, o2, _ = builder()
        r_ca  = evaluate(f"{name} + CA",     wm2, o2, typ_items, o0, ec_base, lam=LAM)
        results.append({"slot": name, "bfs": r_bfs, "ca": r_ca})
        print(f"{name:<20} | BFS: typ {r_bfs['steps_typ']:>4d}/{r_bfs['wpc_typ']:6.2f} "
              f"o0 {r_bfs['steps_o0']:>4d}/{r_bfs['wpc_o0']:6.2f} "
              f"mean={r_bfs['mean_steps']:6.1f}/{r_bfs['mean_wpc']:5.2f} | "
              f"CA:  typ {r_ca['steps_typ']:>4d}/{r_ca['wpc_typ']:6.2f} "
              f"o0 {r_ca['steps_o0']:>4d}/{r_ca['wpc_o0']:6.2f} "
              f"mean={r_ca['mean_steps']:6.1f}/{r_ca['mean_wpc']:5.2f}", flush=True)


    print("\n=== λ-sweep on CAS ===\n", flush=True)
    sweep = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
    sweep_results = []
    for lam in sweep:
        wm, o, _ = build_cas()
        r = evaluate(f"CAS λ={lam}", wm, o, typ_items, o0, ec_base, lam=lam)
        sweep_results.append(r)
        print(f"λ={lam:>4.1f} | typ {r['steps_typ']:>4d}/{r['wpc_typ']:6.2f} "
              f"o0 {r['steps_o0']:>4d}/{r['wpc_o0']:6.2f} "
              f"mean={r['mean_steps']:6.1f}/{r['mean_wpc']:5.2f}", flush=True)


    with open("config_c_part2_results.txt", "w") as f:
        f.write("Config C — Part 2 results (WPC vs ec_base, K=10000, seed=42, λ=5 for CA)\n")
        f.write("=" * 90 + "\n\n")
        f.write("7 SLOTTINGS × 2 ROUTINGS\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Slotting':<20} {'Route':<6} {'typ_st':>6} {'typ_WPC':>8} "
                f"{'o0_st':>6} {'o0_WPC':>8} {'mean_st':>8} {'L95':>4} {'mean_WPC':>9} {'Gini':>8}\n")
        for row in results:
            for tag, r in [("BFS", row["bfs"]), ("CA", row["ca"])]:
                f.write(f"{row['slot']:<20} {tag:<6} {r['steps_typ']:>6d} {r['wpc_typ']:>8.2f} "
                        f"{r['steps_o0']:>6d} {r['wpc_o0']:>8.2f} "
                        f"{r['mean_steps']:>8.1f} {r['L95']:>4d} {r['mean_wpc']:>9.3f} {r['gini']:>8.4f}\n")
        f.write("\nDelta CA vs BFS within each slotting (typical order):\n")
        for row in results:
            ds = pct(row['ca']['steps_typ'], row['bfs']['steps_typ'])
            dw = pct(row['ca']['wpc_typ'],   row['bfs']['wpc_typ'])
            f.write(f"  {row['slot']:<20} typ_st {ds:>7}  typ_WPC {dw:>7}\n")

        f.write("\n\nλ-SWEEP ON CAS\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'λ':>5} {'typ_st':>6} {'typ_WPC':>8} {'o0_st':>6} {'o0_WPC':>8} "
                f"{'mean_st':>8} {'L95':>4} {'mean_WPC':>9} {'Gini':>8}\n")
        for lam, r in zip(sweep, sweep_results):
            f.write(f"{lam:>5.1f} {r['steps_typ']:>6d} {r['wpc_typ']:>8.2f} "
                    f"{r['steps_o0']:>6d} {r['wpc_o0']:>8.2f} "
                    f"{r['mean_steps']:>8.1f} {r['L95']:>4d} {r['mean_wpc']:>9.3f} {r['gini']:>8.4f}\n")
    print("\nSaved: config_c_part2_results.txt", flush=True)


if __name__ == "__main__":
    main()
