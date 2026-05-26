from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple
from collections import deque, defaultdict, Counter
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, PowerNorm
from matplotlib.patches import Patch

from .constants import Cell


class MetricsMixin:
    @staticmethod
    def _gini_coefficient(arr: np.ndarray) -> float:
        """Compute the Gini coefficient of a non-negative array.

        Inputs: numpy array of cell visit counts.
        Returns: float in zero to one; zero means uniform traffic, one means all traffic on one cell.
        """
        vals = np.sort(arr[arr > 0].ravel())
        n = len(vals)
        if n == 0:
            return 0.0
        idx = np.arange(1, n + 1, dtype=float)
        return float((2.0 * (idx * vals).sum() - (n + 1) * vals.sum()) /
                     (n * vals.sum()))

    def build_heatmap(self, orders: List[List[str]]) -> np.ndarray:
        """Simulate every order with BFS routing and accumulate traffic.

        Inputs: list of orders.
        Returns: numpy array heatmap in zero to one. Also stores cell visit counts and edge congestion on the model.
        """
        if self.grid is None:
            raise ValueError("Call generate() first.")

        grid_rows, grid_cols = self.grid.shape
        cell_counts = np.zeros((grid_rows, grid_cols), dtype=float)
        edge_counts: Dict[Tuple, float] = {}

        for idx, order in enumerate(orders):
            path = self._route_order_silent(order)

            for cell in path:
                cell_counts[cell[0], cell[1]] += 1.0

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge = (u, v) if u <= v else (v, u)
                edge_counts[edge] = edge_counts.get(edge, 0.0) + 1.0


        max_cell = cell_counts.max()
        heatmap = (cell_counts / max_cell) if max_cell > 0 else cell_counts.copy()


        if edge_counts:
            max_edge = max(edge_counts.values())
            edge_congestion = {e: v / max_edge for e, v in edge_counts.items()}
        else:
            edge_congestion = {}

        self.cell_counts = cell_counts
        self.heatmap = heatmap
        self.edge_congestion = edge_congestion

        print(f"[build_heatmap] Simulated {len(orders)} orders.")
        print(f"[build_heatmap] Unique edges traversed: {len(edge_congestion)}")
        print(f"[build_heatmap] Max cell visits: {int(max_cell)}, "
              f"Total cell visits: {int(cell_counts.sum())}")

        return heatmap

    def compute_wpc(self, path: List[Cell]) -> float:
        """Compute Weighted Path Congestion for a single path.

        Inputs: list of cells forming the path.
        Returns: float, sum of normalised edge congestion along the path.
        """
        if not self.edge_congestion or not path:
            return 0.0
        total = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            key = (u, v) if u <= v else (v, u)
            total += self.edge_congestion.get(key, 0.0)
        return total

    def compute_crr(
        self,
        bfs_path: List[Cell],
        ca_path: List[Cell],
    ) -> float:
        """Compute Congestion Relief Ratio between baseline and congestion-aware routes.

        Inputs: BFS path, congestion-aware path.
        Returns: float in zero to one, share of baseline cells avoided by the
        congestion-aware route.
        """
        bfs_cells = set(bfs_path)
        ca_cells  = set(ca_path)
        if not bfs_cells:
            return 0.0
        relieved = bfs_cells - ca_cells
        return len(relieved) / len(bfs_cells)

    @staticmethod
    def compute_mean_and_p95(route_lengths: List[int]) -> Tuple[float, float]:
        """Compute mean and 95th percentile of route lengths.

        Inputs: list of route lengths.
        Returns: tuple (mean, p95) as floats.
        """
        if not route_lengths:
            return 0.0, 0.0
        arr = np.array(route_lengths, dtype=float)
        return float(arr.mean()), float(np.percentile(arr, 95))

