from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .constants import Cell, WAREHOUSE_SEED, ORDER_SEED
from .layout import LayoutMixin
from .graph import GraphMixin
from .slotting import SlottingMixin
from .routing import RoutingMixin
from .metrics import MetricsMixin
from .viz import VizMixin


@dataclass
class WarehouseModel(
    LayoutMixin,
    GraphMixin,
    SlottingMixin,
    RoutingMixin,
    MetricsMixin,
    VizMixin,
):
    """Warehouse grid and walkability graph.

    Grid codes:
      0 means walkable aisle.
      1 means shelf, not walkable.
      2 means the single door, walkable, used as entry and exit.
    """
    rows: int = 40
    cols: int = 40

    shelf_length: int = 4
    shelf_gap: int = 1
    shelf_rows_height: int = 2
    aisle_width: int = 2
    margin: int = 1
    shelf_capacity: int = 1

    seed: int = WAREHOUSE_SEED

    door: Cell = (8, 0)

    grid: Optional[np.ndarray] = None
    graph: Optional[Dict[Cell, List[Cell]]] = None

    abc_df: Optional[pd.DataFrame] = None
    shelf_assignments: Optional[Dict[Cell, dict]] = field(default=None, repr=False)
    shelf_cells: Optional[List[Tuple[Cell, float]]] = field(default=None, repr=False)

    heatmap: Optional[np.ndarray] = field(default=None, repr=False)
    cell_counts: Optional[np.ndarray] = field(default=None, repr=False)
    edge_congestion: Optional[Dict] = field(default=None, repr=False)
