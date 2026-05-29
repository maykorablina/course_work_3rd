import importlib
from collections import Counter
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

import warehouse_model
importlib.reload(warehouse_model)
from warehouse_model import WarehouseModel
from central_aisle_slotting import central_aisle_slotting


NUM_ORDERS, ITEMS_PER_ORDER, ORDER_SEED, LAM = 10_000, 15, 42, 5.0


def fresh_wm():
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


def evaluate(label, wm, typ_items, o0, ec_for_wpc, lam):

    if lam == 0.0:
        route = lambda o: wm._route_order_silent(o)
    else:
        route = lambda o: wm._route_order_congestion_silent(o, lam=lam)

    p_typ = route(typ_items)
    p_o0  = route(o0)


    paths = [route(o) for o in wm._orders_cached]
    lengths = [len(p) for p in paths]
    wpcs    = [compute_wpc(p, ec_for_wpc) for p in paths]

    return {
        "label": label,
        "steps_typ": len(p_typ),
        "wpc_typ":   compute_wpc(p_typ, ec_for_wpc),
        "steps_o0":  len(p_o0),
        "wpc_o0":    compute_wpc(p_o0, ec_for_wpc),
        "mean_steps": float(np.mean(lengths)),
        "L95":        int(np.percentile(lengths, 95)),
        "mean_wpc":   float(np.mean(wpcs)),
        "gini":       wm._gini_coefficient(wm.cell_counts),
    }


def print_row(r, base=None, label_w=20):
    if base is None:
        print(f"  {r['label']:<{label_w}} | "
              f"typ: {r['steps_typ']:>4d}st / {r['wpc_typ']:6.2f}W | "
              f"o0:  {r['steps_o0']:>4d}st / {r['wpc_o0']:6.2f}W | "
              f"mean={r['mean_steps']:6.1f}  L95={r['L95']:>4d}  meanW={r['mean_wpc']:5.2f}  Gini={r['gini']:.4f}")
    else:
        ds = pct(r['steps_typ'], base['steps_typ'])
        dw = pct(r['wpc_typ'],   base['wpc_typ'])
        ds0 = pct(r['steps_o0'], base['steps_o0'])
        dw0 = pct(r['wpc_o0'],   base['wpc_o0'])
        dm = pct(r['mean_steps'], base['mean_steps'])
        dW = pct(r['mean_wpc'],  base['mean_wpc'])
        print(f"  {r['label']:<{label_w}} | "
              f"typ: {r['steps_typ']:>4d}st ({ds:>7}) / {r['wpc_typ']:6.2f}W ({dw:>7}) | "
              f"o0:  {r['steps_o0']:>4d}st ({ds0:>7}) / {r['wpc_o0']:6.2f}W ({dw0:>7}) | "
              f"mean={r['mean_steps']:6.1f} ({dm:>7})  L95={r['L95']:>4d}  meanW={r['mean_wpc']:5.2f} ({dW:>7})  Gini={r['gini']:.4f}")


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

    print("Building Baseline (ABC×XYZ)...")
    wm_base, orders_base = fresh_wm()
    wm_base._orders_cached = orders_base
    ec_base = wm_base.edge_congestion

    typ_items = [it for it, _ in
                 Counter(it for o in orders_base for it in o).most_common(ITEMS_PER_ORDER)]
    o0 = orders_base[0]


    print("Building CAS-slotted model...")
    wm_cas, orders_cas = fresh_wm()
    central_aisle_slotting(wm_cas, orders_cas)
    wm_cas._orders_cached = orders_cas
    ec_cas = wm_cas.edge_congestion


    print("\n=== Four configurations (WPC measured against Baseline ec_base) ===\n")

    baseline = evaluate("Baseline (B+BFS)",   wm_base, typ_items, o0, ec_base, lam=0.0)
    print_row(baseline, label_w=22)

    config_a = evaluate("Config A (B+CA)",    wm_base, typ_items, o0, ec_base, lam=LAM)
    print_row(config_a, base=baseline, label_w=22)

    config_b_cas = evaluate("Config B-CAS",   wm_cas,  typ_items, o0, ec_base, lam=0.0)
    print_row(config_b_cas, base=baseline, label_w=22)

    config_c = evaluate("Config C (CAS+CA)", wm_cas,  typ_items, o0, ec_base, lam=LAM)
    print_row(config_c, base=baseline, label_w=22)


    print("\n=== Config C deltas ===\n")
    print(f"vs Baseline:           typ {pct(config_c['steps_typ'], baseline['steps_typ']):>7} steps, "
          f"{pct(config_c['wpc_typ'], baseline['wpc_typ']):>7} WPC | "
          f"o0 {pct(config_c['steps_o0'], baseline['steps_o0']):>7} steps, "
          f"{pct(config_c['wpc_o0'], baseline['wpc_o0']):>7} WPC")
    print(f"vs Config A (CA only): typ {pct(config_c['steps_typ'], config_a['steps_typ']):>7} steps, "
          f"{pct(config_c['wpc_typ'], config_a['wpc_typ']):>7} WPC | "
          f"o0 {pct(config_c['steps_o0'], config_a['steps_o0']):>7} steps, "
          f"{pct(config_c['wpc_o0'], config_a['wpc_o0']):>7} WPC")
    print(f"vs B-CAS  (CAS only):  typ {pct(config_c['steps_typ'], config_b_cas['steps_typ']):>7} steps, "
          f"{pct(config_c['wpc_typ'], config_b_cas['wpc_typ']):>7} WPC | "
          f"o0 {pct(config_c['steps_o0'], config_b_cas['steps_o0']):>7} steps, "
          f"{pct(config_c['wpc_o0'], config_b_cas['wpc_o0']):>7} WPC")


    with open("config_c_part1_results.txt", "w") as f:
        f.write("Config C — Part 1: CAS slotting + Dijkstra-NN routing\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"{'Config':<22} {'typ_st':>6} {'typ_WPC':>8} {'o0_st':>6} {'o0_WPC':>8} "
                f"{'mean_st':>8} {'L95':>4} {'mean_WPC':>9} {'Gini':>8}\n")
        for r in [baseline, config_a, config_b_cas, config_c]:
            f.write(f"{r['label']:<22} {r['steps_typ']:>6d} {r['wpc_typ']:>8.2f} "
                    f"{r['steps_o0']:>6d} {r['wpc_o0']:>8.2f} "
                    f"{r['mean_steps']:>8.1f} {r['L95']:>4d} {r['mean_wpc']:>9.3f} {r['gini']:>8.4f}\n")

        f.write("\nDelta against Baseline:\n")
        for r in [config_a, config_b_cas, config_c]:
            f.write(f"  {r['label']:<22} typ {pct(r['steps_typ'], baseline['steps_typ']):>7} steps "
                    f"{pct(r['wpc_typ'], baseline['wpc_typ']):>7} WPC | "
                    f"o0 {pct(r['steps_o0'], baseline['steps_o0']):>7} steps "
                    f"{pct(r['wpc_o0'], baseline['wpc_o0']):>7} WPC | "
                    f"mean {pct(r['mean_steps'], baseline['mean_steps']):>7} "
                    f"{pct(r['mean_wpc'], baseline['mean_wpc']):>7} WPC\n")
    print("\nSaved: config_c_part1_results.txt")


    print("\nBuilding Config C heatmap from Dijkstra-NN routes...")
    rows_, cols_ = wm_cas.grid.shape
    cell_counts_c = np.zeros((rows_, cols_), dtype=np.int64)
    for o in orders_cas:
        path = wm_cas._route_order_congestion_silent(o, lam=LAM)
        for cell in path:
            cell_counts_c[cell[0], cell[1]] += 1
    wm_cas_c = wm_cas
    wm_cas_c.cell_counts = cell_counts_c


    print("Building Config A heatmap from Dijkstra-NN routes...")
    cell_counts_a = np.zeros((rows_, cols_), dtype=np.int64)
    for o in orders_base:
        path = wm_base._route_order_congestion_silent(o, lam=LAM)
        for cell in path:
            cell_counts_a[cell[0], cell[1]] += 1
    wm_base.cell_counts = cell_counts_a

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    draw_heatmap_panel(axes[0], wm_base,   "Config A (Baseline slotting + Dijkstra-NN)")
    draw_heatmap_panel(axes[1], wm_cas_c,  "Config C (CAS slotting + Dijkstra-NN)")
    fig.suptitle("Config A vs Config C — heatmap of Dijkstra-NN traffic",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = "final/fig_config_c_part1_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
