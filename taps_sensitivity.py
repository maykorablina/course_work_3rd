import matplotlib
matplotlib.use("Agg")

import importlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from collections import Counter

import warehouse_model
importlib.reload(warehouse_model)
from warehouse_model import WarehouseModel


NUM_ORDERS, ITEMS_PER_ORDER, ORDER_SEED = 10_000, 15, 42


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


def run_taps(alpha, baseline_ec, typ_items, o0):
    wm, _ = make_wm()
    orders_local = wm.generate_orders(num_orders=NUM_ORDERS,
                                      items_per_order=ITEMS_PER_ORDER,
                                      order_seed=ORDER_SEED,
                                      weight_col="Total_Annual_Units", skew=1.0)
    r = wm.taps_slotting(orders_local, alpha=alpha, rebuild=True, verbose=False)
    p_typ = wm._route_order_silent(typ_items)
    p_o0  = wm._route_order_silent(o0)
    return {
        "alpha": alpha,
        "gini_after": r["gini_after"],
        "wpc_typ": compute_wpc(p_typ, baseline_ec),
        "steps_typ": len(p_typ),
        "wpc_o0":  compute_wpc(p_o0, baseline_ec),
        "steps_o0": len(p_o0),
    }, wm


def main():
    wm_base, orders_base = make_wm()
    typ_items = [it for it, _ in Counter(it for o in orders_base for it in o).most_common(ITEMS_PER_ORDER)]
    o0 = orders_base[0]

    p_typ_b = wm_base._route_order_silent(typ_items)
    p_o0_b  = wm_base._route_order_silent(o0)
    base_steps_typ = len(p_typ_b)
    base_wpc_typ   = compute_wpc(p_typ_b, wm_base.edge_congestion)
    base_steps_o0  = len(p_o0_b)
    base_wpc_o0    = compute_wpc(p_o0_b, wm_base.edge_congestion)
    base_gini      = wm_base._gini_coefficient(wm_base.cell_counts)
    print(f"Baseline | typ: steps={base_steps_typ}, WPC={base_wpc_typ:.2f} | "
          f"o0: steps={base_steps_o0}, WPC={base_wpc_o0:.2f} | Gini={base_gini:.4f}")

    sweep = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    rows = []
    wm_alpha3 = None
    for a in sweep:
        r, wm_a = run_taps(a, wm_base.edge_congestion, typ_items, o0)
        r["d_wpc_typ_pct"]   = (r["wpc_typ"]   - base_wpc_typ)   / base_wpc_typ   * 100
        r["d_steps_typ_pct"] = (r["steps_typ"] - base_steps_typ) / base_steps_typ * 100
        r["d_wpc_o0_pct"]    = (r["wpc_o0"]    - base_wpc_o0)    / base_wpc_o0    * 100
        r["d_steps_o0_pct"]  = (r["steps_o0"]  - base_steps_o0)  / base_steps_o0  * 100
        rows.append(r)
        print(f"alpha={a:>5.1f} | Gini={r['gini_after']:.4f} | "
              f"typ: steps={r['steps_typ']:3d} ({r['d_steps_typ_pct']:+.0f}%) "
              f"WPC={r['wpc_typ']:5.2f} ({r['d_wpc_typ_pct']:+.0f}%) | "
              f"o0: steps={r['steps_o0']:3d} ({r['d_steps_o0_pct']:+.0f}%) "
              f"WPC={r['wpc_o0']:5.2f} ({r['d_wpc_o0_pct']:+.0f}%)")
        if abs(a - 3.0) < 1e-6:
            wm_alpha3 = wm_a


    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    xs = [r["alpha"] for r in rows]
    wpcs_typ   = [r["wpc_typ"] for r in rows]
    steps_typ  = [r["steps_typ"] for r in rows]
    ginis      = [r["gini_after"] for r in rows]

    ax = axes[0]
    ax.plot(xs, wpcs_typ, "o-", color="#c0392b", lw=2, ms=7, label="Typical order")
    ax.plot(xs, [r["wpc_o0"] for r in rows], "s--", color="#8e44ad", lw=1.6, ms=6, label="Order o_0")
    ax.axhline(base_wpc_typ, color="#c0392b", ls=":", lw=1.0, alpha=0.6, label=f"Baseline typ = {base_wpc_typ:.1f}")
    ax.axhline(base_wpc_o0,  color="#8e44ad", ls=":", lw=1.0, alpha=0.6, label=f"Baseline o_0 = {base_wpc_o0:.1f}")
    ax.axvline(3.0, color="#27ae60", ls="--", lw=1.2, alpha=0.7)
    ax.text(3.0, ax.get_ylim()[1]*0.95 if False else max(wpcs_typ)*1.05,
            r"$\alpha=3$", color="#27ae60", fontsize=9, ha="center")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("WPC")
    ax.set_title("Weighted Path Congestion")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.4)

    ax = axes[1]
    ax.plot(xs, steps_typ, "o-", color="#2980b9", lw=2, ms=7, label="Typical order")
    ax.plot(xs, [r["steps_o0"] for r in rows], "s--", color="#16a085", lw=1.6, ms=6, label="Order o_0")
    ax.axhline(base_steps_typ, color="#2980b9", ls=":", lw=1.0, alpha=0.6, label=f"Baseline typ = {base_steps_typ}")
    ax.axhline(base_steps_o0,  color="#16a085", ls=":", lw=1.0, alpha=0.6, label=f"Baseline o_0 = {base_steps_o0}")
    ax.axvline(3.0, color="#27ae60", ls="--", lw=1.2, alpha=0.7)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Route length (steps)")
    ax.set_title("Route Length")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.4)

    ax = axes[2]
    ax.plot(xs, ginis, "^-", color="#27ae60", lw=2, ms=7)
    ax.axhline(base_gini, color="#888", ls="--", lw=1.2, label=f"Baseline = {base_gini:.3f}")
    ax.axvline(3.0, color="#27ae60", ls="--", lw=1.2, alpha=0.5)
    for x, y in zip(xs, ginis):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Gini")
    ax.set_title("Traffic Gini (lower = more uniform)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.4)

    fig.suptitle(r"TAPS sensitivity to $\alpha$ (K=10000, baseline congestion field)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = "final/fig_taps_sensitivity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    for ax, (title, model) in zip(axes, [
        ("Config A: ABC×XYZ Baseline", wm_base),
        ("Config B: TAPS Slotting (α=3)", wm_alpha3),
    ]):
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
    fig.suptitle("Config A vs TAPS (α=3) — Traffic Heatmap",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out2 = "final/fig_taps_heatmap.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")

    with open("taps_sensitivity_results.txt", "w") as f:
        f.write(f"Baseline | typ: steps={base_steps_typ} WPC={base_wpc_typ:.2f} | "
                f"o0: steps={base_steps_o0} WPC={base_wpc_o0:.2f} | Gini={base_gini:.4f}\n\n")
        f.write(f"{'alpha':>6} {'Gini':>8} | {'tp_steps':>8} {'d_st_t%':>8} {'tp_WPC':>8} {'d_W_t%':>8} | "
                f"{'o0_steps':>8} {'d_st_0%':>8} {'o0_WPC':>8} {'d_W_0%':>8}\n")
        for r in rows:
            f.write(f"{r['alpha']:>6.1f} {r['gini_after']:>8.4f} | "
                    f"{r['steps_typ']:>8d} {r['d_steps_typ_pct']:>+7.1f}% "
                    f"{r['wpc_typ']:>8.2f} {r['d_wpc_typ_pct']:>+7.1f}% | "
                    f"{r['steps_o0']:>8d} {r['d_steps_o0_pct']:>+7.1f}% "
                    f"{r['wpc_o0']:>8.2f} {r['d_wpc_o0_pct']:>+7.1f}%\n")
    print("Saved: taps_sensitivity_results.txt")


if __name__ == "__main__":
    main()
