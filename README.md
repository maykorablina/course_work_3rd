# course_work_3rd

Term project for the 3rd year of the Bachelor's Programme "Applied Mathematics
and Informatics" at HSE Faculty of Computer Science.

The work studies warehouse order picking optimisation through the integration
of ABC x XYZ inventory classification, heat-map-based congestion modelling,
and graph-based routing.

## Repository structure

- `test_library/` — a compact Python package that implements the core of the
  proposed framework: warehouse layout generation, walkability graph, BFS
  routing, baseline ABC x XYZ slotting, heat-map construction, and Weighted
  Path Congestion (WPC) and route-length metrics.
- `example/example_usage.py` — minimal script that runs the full pipeline:
  build the layout, classify the SKUs, place them on shelves, generate orders,
  build a heat map, and compute route length and WPC.

## Quick start

```bash
python example/example_usage.py
```

The example expects an SKU table at `data/abc_xyz_dataset.csv`. The dataset
used in the experiments is the public Kaggle "ABC-XYZ Inventory
Classification Dataset".

## Library modules

- `test_library/constants.py` — type aliases and random seeds.
- `test_library/layout.py` — warehouse grid generation.
- `test_library/graph.py` — walkability graph and BFS shortest-path search.
- `test_library/slotting.py` — baseline ABC x XYZ slotting.
- `test_library/routing.py` — order generation and BFS routing.
- `test_library/metrics.py` — heat map construction, WPC, mean and p95
  route length.
- `test_library/model.py` — the WarehouseModel class that composes all
  mixins above.
