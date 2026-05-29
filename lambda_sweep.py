import numpy as np
import sys
from collections import Counter
sys.path.insert(0, '.')
from warehouse_model import WarehouseModel


wm = WarehouseModel(
    rows=52, cols=52,
    shelf_length=4, shelf_gap=1,
    shelf_rows_height=2, aisle_width=2,
    margin=1, shelf_capacity=1,
    seed=6, door=(23, 0),
)
wm.generate()
wm.build_graph()
wm.abc_classify(
    csv_path='abc_xyz_dataset.csv',
    a_pct=0.20, b_pct=0.30,
    x_cv=0.10, y_cv=0.25,
)
wm.assign_items_to_shelves()

NUM_ORDERS      = 10000
ITEMS_PER_ORDER = 15
ORDER_SEED      = 42

orders = wm.generate_orders(
    num_orders=NUM_ORDERS,
    items_per_order=ITEMS_PER_ORDER,
    order_seed=ORDER_SEED,
    weight_col='Total_Annual_Units',
    skew=1.0,
)
print(f'Generated {len(orders)} orders', flush=True)

wm.build_heatmap(orders)
print('Heatmap built', flush=True)


bfs_paths = [wm._route_order_silent(o) for o in orders]
bfs_lengths = [len(p) for p in bfs_paths]
bfs_wpc_list = [wm.compute_wpc(p) for p in bfs_paths]
bfs_mean = float(np.mean(bfs_lengths))
bfs_L95  = int(np.percentile(bfs_lengths, 95))
bfs_wpc_mean = float(np.mean(bfs_wpc_list))
print(f'Baseline (λ=0): mean_L={bfs_mean:.1f}  L95={bfs_L95}  mean_WPC={bfs_wpc_mean:.1f}', flush=True)


item_freq = Counter(item for order in orders for item in order)
typical_order = [item for item, _ in item_freq.most_common(ITEMS_PER_ORDER)]

typ_bfs_path = wm._route_order_silent(typical_order)
typ_bfs_wpc  = wm.compute_wpc(typ_bfs_path)
typ_bfs_len  = len(typ_bfs_path)
print(f'Typical order BFS: len={typ_bfs_len}  WPC={typ_bfs_wpc:.2f}', flush=True)


bfs_cell_sets = [set(p) for p in bfs_paths]


lambdas = [1, 2, 5, 6, 7, 8, 10, 20]

print('\n' + '='*95, flush=True)
print(f'{"lam":>5}  {"mean_L":>8}  {"dL%":>7}  {"L95":>6}  {"dL95%":>7}  '
      f'{"mean_WPC":>9}  {"dWPC%":>7}  {"mean_CRR":>9}  {"typ_len":>8}  {"typ_WPC":>9}  {"dTypWPC%":>9}')
print('-'*95)

for lam in lambdas:
    ca_lengths = []
    ca_wpc_list = []
    crr_list = []

    for i, o in enumerate(orders):
        path = wm._route_order_congestion_silent(o, lam=lam)
        ca_lengths.append(len(path))
        ca_wpc_list.append(wm.compute_wpc(path))

        if bfs_cell_sets[i]:
            avoided = len(bfs_cell_sets[i] - set(path))
            crr_list.append(avoided / len(bfs_cell_sets[i]))
        else:
            crr_list.append(0.0)

    ca_mean = float(np.mean(ca_lengths))
    ca_L95  = int(np.percentile(ca_lengths, 95))
    ca_wpc_mean = float(np.mean(ca_wpc_list))
    ca_crr_mean = float(np.mean(crr_list))


    typ_path = wm._route_order_congestion_silent(typical_order, lam=lam)
    typ_len  = len(typ_path)
    typ_wpc  = wm.compute_wpc(typ_path)

    dL_pct    = (ca_mean - bfs_mean) / bfs_mean * 100
    dL95_pct  = (ca_L95  - bfs_L95)  / bfs_L95  * 100
    dWPC_pct  = (ca_wpc_mean - bfs_wpc_mean) / bfs_wpc_mean * 100
    dTypWPC_pct = (typ_wpc - typ_bfs_wpc) / typ_bfs_wpc * 100

    print(f'{lam:>5}  {ca_mean:>8.1f}  {dL_pct:>+6.0f}%  {ca_L95:>6}  {dL95_pct:>+6.0f}%  '
          f'{ca_wpc_mean:>9.1f}  {dWPC_pct:>+6.0f}%  {ca_crr_mean:>9.3f}  '
          f'{typ_len:>8}  {typ_wpc:>9.2f}  {dTypWPC_pct:>+8.0f}%', flush=True)

print('='*95)
print(f'Baseline row: mean_L={bfs_mean:.1f}  L95={bfs_L95}  mean_WPC={bfs_wpc_mean:.1f}  '
      f'typ_len={typ_bfs_len}  typ_WPC={typ_bfs_wpc:.2f}')
