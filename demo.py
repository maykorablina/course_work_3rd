import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from warehouse_model import WarehouseModel


SEED = 42
K    = 10_000
N    = 15
LAM  = 5.0

np.random.seed(SEED)


print("=== Building warehouse ===")
wm = WarehouseModel(rows=52, cols=52, seed=6, door=(23, 0))
wm.generate()
wm.build_graph()

wm.abc_classify("abc_xyz_dataset.csv")
wm.assign_items_to_shelves()


print(f"\n=== Generating {K} orders ({N} items each) ===")
orders = wm.generate_orders(num_orders=K, items_per_order=N, order_seed=SEED)

K_TRAIN = K // 2
orders_train = orders[:K_TRAIN]
orders_test  = orders[K_TRAIN:]
print(f"  Train (heatmap): {len(orders_train)}  | Test (evaluation): {len(orders_test)}")


print("\n=== Building baseline heatmap (BFS, λ=0) on TRAIN orders ===")
wm.build_heatmap(orders_train)


print(f"\n=== Batch BFS routing ({len(orders_test)} TEST orders) — Baseline ===")
bfs_lengths = wm.batch_route_lengths(orders_test, lam=0.0)
bfs_mean, bfs_L95 = WarehouseModel.compute_mean_and_p95(bfs_lengths)


bfs_wpc_list = []
for order in orders_test:
    path = wm._route_order_silent(order)
    bfs_wpc_list.append(wm.compute_wpc(path))
bfs_wpc_mean = float(np.mean(bfs_wpc_list))

print(f"  Baseline  | Mean steps: {bfs_mean:.1f}  | L95: {bfs_L95:.0f}  | Mean WPC: {bfs_wpc_mean:.3f}")


print(f"\n=== Batch CA routing ({len(orders_test)} TEST orders) — Config A (λ={LAM}) ===")
ca_lengths = wm.batch_route_lengths(orders_test, lam=LAM)
ca_mean, ca_L95 = WarehouseModel.compute_mean_and_p95(ca_lengths)


ca_wpc_list = []
for order in orders_test:
    path = wm._route_order_congestion_silent(order, lam=LAM)
    ca_wpc_list.append(wm.compute_wpc(path))
ca_wpc_mean = float(np.mean(ca_wpc_list))

print(f"  Config A  | Mean steps: {ca_mean:.1f}  | L95: {ca_L95:.0f}  | Mean WPC: {ca_wpc_mean:.3f}")


print("\n" + "=" * 65)
print(f"{'Metric':<30} {'Baseline':>12} {'Config A':>12} {'Δ':>9}")
print("-" * 65)

d_mean   = ca_mean     - bfs_mean
d_L95    = ca_L95      - bfs_L95
d_wpc    = ca_wpc_mean - bfs_wpc_mean

def pct(delta, base):
    return f"({delta/base*100:+.1f}%)" if base else ""

print(f"{'Mean route length (steps)':<30} {bfs_mean:>12.1f} {ca_mean:>12.1f} {d_mean:>+8.1f} {pct(d_mean, bfs_mean)}")
print(f"{'L95 (95th percentile steps)':<30} {bfs_L95:>12.0f} {ca_L95:>12.0f} {d_L95:>+8.0f} {pct(d_L95, bfs_L95)}")
print(f"{'Mean WPC':<30} {bfs_wpc_mean:>12.3f} {ca_wpc_mean:>12.3f} {d_wpc:>+8.3f} {pct(d_wpc, bfs_wpc_mean)}")
print("=" * 65)


print("\n=== CRR for two representative orders ===")


order_o0 = orders[0]
bfs_path_o0 = wm._route_order_silent(order_o0)
ca_path_o0  = wm._route_order_congestion_silent(order_o0, lam=LAM)
crr_o0 = wm.compute_crr(bfs_path_o0, ca_path_o0)

print(f"  Order o_0  | BFS steps: {len(bfs_path_o0):4d}  CA steps: {len(ca_path_o0):4d}"
      f"  WPC_bfs: {wm.compute_wpc(bfs_path_o0):.2f}"
      f"  WPC_ca: {wm.compute_wpc(ca_path_o0):.2f}"
      f"  CRR: {crr_o0:.2%}")


from collections import Counter
item_counts: Counter = Counter()
for order in orders:
    item_counts.update(order)
typical_items = [item for item, _ in item_counts.most_common(N)]

bfs_path_typ = wm._route_order_silent(typical_items)
ca_path_typ  = wm._route_order_congestion_silent(typical_items, lam=LAM)
crr_typ = wm.compute_crr(bfs_path_typ, ca_path_typ)

print(f"  Typical    | BFS steps: {len(bfs_path_typ):4d}  CA steps: {len(ca_path_typ):4d}"
      f"  WPC_bfs: {wm.compute_wpc(bfs_path_typ):.2f}"
      f"  WPC_ca: {wm.compute_wpc(ca_path_typ):.2f}"
      f"  CRR: {crr_typ:.2%}")


fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(
    f"Baseline vs Config A (λ={LAM}) — K={K:,} orders",
    fontsize=13, fontweight="bold"
)


ax = axes[0]
bins = np.linspace(
    min(min(bfs_lengths), min(ca_lengths)),
    max(max(bfs_lengths), max(ca_lengths)),
    40
)
ax.hist(bfs_lengths, bins=bins, alpha=0.55, color="#e67e22", label="Baseline (BFS)")
ax.hist(ca_lengths,  bins=bins, alpha=0.55, color="#2980b9", label=f"Config A (λ={LAM})")
ax.axvline(bfs_mean, color="#e67e22", linestyle="--", lw=1.5, label=f"μ Baseline={bfs_mean:.0f}")
ax.axvline(ca_mean,  color="#2980b9", linestyle="--", lw=1.5, label=f"μ Config A={ca_mean:.0f}")
ax.axvline(bfs_L95, color="#e67e22", linestyle=":", lw=1.5, label=f"L95 Baseline={bfs_L95:.0f}")
ax.axvline(ca_L95,  color="#2980b9", linestyle=":", lw=1.5, label=f"L95 Config A={ca_L95:.0f}")
ax.set_xlabel("Route length (steps)")
ax.set_ylabel("Order count")
ax.set_title("Route Length Distribution")
ax.legend(fontsize=7)


ax = axes[1]
bins_wpc = np.linspace(
    min(min(bfs_wpc_list), min(ca_wpc_list)),
    max(max(bfs_wpc_list), max(ca_wpc_list)),
    40
)
ax.hist(bfs_wpc_list, bins=bins_wpc, alpha=0.55, color="#e67e22", label="Baseline (BFS)")
ax.hist(ca_wpc_list,  bins=bins_wpc, alpha=0.55, color="#2980b9", label=f"Config A (λ={LAM})")
ax.axvline(bfs_wpc_mean, color="#e67e22", linestyle="--", lw=1.5)
ax.axvline(ca_wpc_mean,  color="#2980b9", linestyle="--", lw=1.5)
ax.set_xlabel("WPC = Σ c_e")
ax.set_ylabel("Order count")
ax.set_title("WPC Distribution")
ax.legend(fontsize=7)


ax = axes[2]
metrics_labels = ["Mean steps", "L95 steps", "Mean WPC"]
bfs_vals = [bfs_mean, bfs_L95, bfs_wpc_mean]
ca_vals  = [ca_mean,  ca_L95,  ca_wpc_mean]

x = np.arange(len(metrics_labels))
w = 0.35
b1 = ax.bar(x - w/2, bfs_vals, w, color="#e67e22", label="Baseline")
b2 = ax.bar(x + w/2, ca_vals,  w, color="#2980b9", label=f"Config A (λ={LAM})")
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels, fontsize=9)
ax.set_title("Metrics Comparison")
ax.legend(fontsize=8)
for bar in list(b1) + list(b2):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{bar.get_height():.1f}",
        ha="center", va="bottom", fontsize=7
    )

plt.tight_layout()
plt.savefig("fig_configA_metrics_full.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: fig_configA_metrics_full.png")

