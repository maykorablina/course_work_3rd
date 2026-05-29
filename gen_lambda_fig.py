import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import re
from collections import Counter


def load_from_file(path="lambda_sweep_results.txt"):

    try:
        with open(path) as f:
            text = f.read()
    except FileNotFoundError:
        return None, None, None, None


    m_base = re.search(r'Baseline.*?mean_L=([\d.]+).*?L95=(\d+).*?mean_WPC=([\d.]+).*?typ_len=(\d+).*?typ_WPC=([\d.]+)',
                       text)
    if not m_base:
        return None, None, None, None

    bfs_mean_L = float(m_base.group(1))
    bfs_wpc    = float(m_base.group(3))


    lambdas, mean_Ls, mean_WPCs = [], [], []
    for line in text.split('\n'):

        m = re.match(r'\s{1,4}(\d+)\s+([\d.]+)\s+', line)
        if m:
            lam = int(m.group(1))
            parts = line.split()
            if len(parts) >= 6:
                try:
                    lambdas.append(lam)
                    mean_Ls.append(float(parts[1]))
                    mean_WPCs.append(float(parts[5]))
                except (ValueError, IndexError):
                    pass

    if len(lambdas) < 3:
        return None, None, None, None

    return lambdas, mean_Ls, mean_WPCs, (bfs_mean_L, bfs_wpc)


def compute_inline():

    from warehouse_model import WarehouseModel

    NUM_ORDERS, ITEMS_PER_ORDER, ORDER_SEED = 10_000, 15, 42
    wm = WarehouseModel(rows=52, cols=52, shelf_length=4, shelf_gap=1,
                        shelf_rows_height=2, aisle_width=2, margin=1,
                        shelf_capacity=1, seed=6, door=(23, 0))
    wm.generate(); wm.build_graph()
    wm.abc_classify(csv_path='abc_xyz_dataset.csv',
                    a_pct=0.20, b_pct=0.30, x_cv=0.10, y_cv=0.25)
    wm.assign_items_to_shelves()
    orders = wm.generate_orders(num_orders=NUM_ORDERS, items_per_order=ITEMS_PER_ORDER,
                                order_seed=ORDER_SEED, weight_col='Total_Annual_Units', skew=1.0)
    wm.build_heatmap(orders)

    bfs_paths = [wm._route_order_silent(o) for o in orders]
    bfs_lengths = [len(p) for p in bfs_paths]
    bfs_wpc_list = [wm.compute_wpc(p) for p in bfs_paths]
    bfs_mean_L = float(np.mean(bfs_lengths))
    bfs_wpc    = float(np.mean(bfs_wpc_list))

    lambdas = [1, 2, 5, 6, 7, 8, 10, 20]
    mean_Ls, mean_WPCs = [], []
    for lam in lambdas:
        paths = [wm._route_order_congestion_silent(o, lam=lam) for o in orders]
        mean_Ls.append(float(np.mean([len(p) for p in paths])))
        mean_WPCs.append(float(np.mean([wm.compute_wpc(p) for p in paths])))
        print(f"  λ={lam}: mean_L={mean_Ls[-1]:.1f}  mean_WPC={mean_WPCs[-1]:.2f}", flush=True)

    return lambdas, mean_Ls, mean_WPCs, (bfs_mean_L, bfs_wpc)


print("Loading lambda sweep results...", flush=True)
lambdas, mean_Ls, mean_WPCs, baseline = load_from_file()

if lambdas is None:
    print("lambda_sweep_results.txt not found or incomplete — computing inline.", flush=True)
    lambdas, mean_Ls, mean_WPCs, baseline = compute_inline()

bfs_mean_L, bfs_wpc = baseline
print(f"Baseline BFS: mean_L={bfs_mean_L:.1f}, mean_WPC={bfs_wpc:.2f}")
print(f"Lambda points: {list(zip(lambdas, [round(w,2) for w in mean_WPCs]))}")

color      = '#2980b9'
base_color = '#c0392b'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle(r'Congestion-Aware Routing: sensitivity to $\lambda$'
             ' (K = 10,000 orders, demand-weighted)',
             fontsize=12, fontweight='bold')


ax1.plot(lambdas, mean_WPCs, 'o-', color=color, linewidth=2,
         markersize=7, zorder=3, label=r'CA routing ($\lambda > 0$)')
ax1.axhline(bfs_wpc, color=base_color, linestyle='--', linewidth=1.8,
            label=f'Baseline BFS (WPC = {bfs_wpc:.1f})', zorder=2)

idx5 = lambdas.index(5) if 5 in lambdas else None
if idx5 is not None:
    ax1.axvline(5, color='#27ae60', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.annotate(r'$\lambda = 5$ (selected)', xy=(5, mean_WPCs[idx5]),
                 xytext=(8, mean_WPCs[idx5] + bfs_wpc * 0.08), fontsize=8,
                 arrowprops=dict(arrowstyle='->', color='#27ae60'),
                 color='#27ae60')
ax1.set_xlabel(r'$\lambda$', fontsize=12)
ax1.set_ylabel('Mean WPC', fontsize=11)
ax1.set_title('Mean Weighted Path Congestion', fontsize=10)
ax1.set_xticks(lambdas)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)


ax2.plot(lambdas, mean_Ls, 's-', color=color, linewidth=2,
         markersize=7, zorder=3, label=r'CA routing ($\lambda > 0$)')
ax2.axhline(bfs_mean_L, color=base_color, linestyle='--', linewidth=1.8,
            label=f'Baseline ({bfs_mean_L:.1f} steps)', zorder=2)
if idx5 is not None:
    ax2.axvline(5, color='#27ae60', linestyle=':', linewidth=1.5, alpha=0.7)
ax2.set_xlabel(r'$\lambda$', fontsize=12)
ax2.set_ylabel('Mean route length (steps)', fontsize=11)
ax2.set_title('Mean Route Length', fontsize=10)
ax2.set_xticks(lambdas)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out = 'final/fig_lambda_sensitivity.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')
