"""Example usage of test_library.

This script demonstrates the standard pipeline of the test_library:
  1. Build the warehouse grid and walkability graph.
  2. Load the SKU dataset and apply ABC x XYZ classification.
  3. Place items on shelves with the baseline ABC x XYZ slotting.
  4. Generate a synthetic order set with demand-weighted sampling.
  5. Build the traffic heat map by simulating all orders.
  6. Compute route length and Weighted Path Congestion (WPC) for one order.

Run from the repository root:
    python example/example_usage.py
"""
from __future__ import annotations

import os
import sys

# Allow running the script from the repo root without installing the package
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from test_library import WarehouseModel


# Path to the SKU table (Kaggle ABC-XYZ Inventory Classification Dataset)
CSV_PATH = os.path.join(ROOT, "data", "abc_xyz_dataset.csv")


def main() -> None:
    # 1. Build the warehouse grid and walkability graph
    model = WarehouseModel(
        rows=40,
        cols=40,
        door=(8, 0),
    )
    model.generate()
    model.build_graph()
    print(f"Grid size: {model.grid.shape}")
    print(f"Walkable cells in the graph: {len(model.graph)}")

    # 2. Load SKU data and apply ABC x XYZ classification
    if not os.path.exists(CSV_PATH):
        print(f"\nSKU dataset not found at {CSV_PATH}.")
        print("Please place 'abc_xyz_dataset.csv' under data/ to continue.")
        return

    model.abc_classify(CSV_PATH)
    print(f"\nClassified {len(model.abc_df)} SKUs.")
    print(model.summary_table().head())

    # 3. Assign items to shelves with baseline ABC x XYZ slotting
    assignments = model.assign_items_to_shelves()
    print(f"\nPlaced items on {len(assignments)} shelves.")

    # 4. Generate a synthetic set of orders
    orders = model.generate_orders(
        num_orders=1000,
        items_per_order=15,
        order_seed=42,
    )

    # 5. Build the traffic heat map
    model.build_heatmap(orders)

    # 6. Route a single order and compute basic metrics
    sample_order = orders[0]
    path = model._route_order_silent(sample_order)
    wpc = model.compute_wpc(path)
    print(f"\nSample order route length: {len(path)} steps")
    print(f"Sample order WPC: {wpc:.4f}")

    # Batch metrics over all orders
    lengths = model.batch_route_lengths(orders)
    mean_len, p95_len = model.compute_mean_and_p95(lengths)
    print(f"\nBatch route length over {len(orders)} orders:")
    print(f"  mean = {mean_len:.1f} steps")
    print(f"  p95  = {p95_len:.1f} steps")


if __name__ == "__main__":
    main()
