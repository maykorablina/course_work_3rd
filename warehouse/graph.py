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


class GraphMixin:
    def build_graph(self, walkable_values: Iterable[int] = (0, 2)) -> Dict[Cell, List[Cell]]:
        """Build the walkability graph from the grid.

        Inputs: walkable_values, iterable of grid codes treated as walkable.
        Returns: adjacency list mapping each walkable cell to its 4-neighbours.
        """
        if self.grid is None:
            raise ValueError("Grid is not generated. Call generate() first.")
        grid = self.grid
        rows, cols = grid.shape
        walkable_set = set(walkable_values)

        def is_walkable(r: int, c: int) -> bool:
            return grid[r, c] in walkable_set

        g: Dict[Cell, List[Cell]] = {}
        for r in range(rows):
            for c in range(cols):
                if not is_walkable(r, c):
                    continue
                node = (r, c)
                neigh: List[Cell] = []
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < rows and 0 <= cc < cols and is_walkable(rr, cc):
                        neigh.append((rr, cc))
                g[node] = neigh

        self.graph = g
        return g

    def shortest_path_bfs(self, start: Cell, goal: Cell) -> Optional[List[Cell]]:
        """Find the shortest path in steps between two cells using BFS.

        Inputs: start cell, goal cell.
        Returns: list of cells from start to goal, or None if unreachable.
        """
        if self.graph is None:
            self.build_graph()

        g = self.graph
        if start not in g or goal not in g:
            return None

        q = deque([start])
        parent: Dict[Cell, Optional[Cell]] = {start: None}

        while q:
            v = q.popleft()
            if v == goal:
                break
            for u in g[v]:
                if u not in parent:
                    parent[u] = v
                    q.append(u)

        if goal not in parent:
            return None


        path: List[Cell] = []
        cur: Optional[Cell] = goal
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    def corridor_distances(self, door: Optional[Cell] = None) -> np.ndarray:
        """Compute BFS distance from the door to every walkable cell.

        Inputs: optional door cell, otherwise the model default is used.
        Returns: numpy array of distances, with infinity for unreachable cells.
        """
        if self.grid is None:
            raise ValueError("Grid is not generated. Call generate() first.")
        grid = self.grid
        rows, cols = grid.shape
        d = door if door is not None else self.door

        dist = np.full((rows, cols), np.inf, dtype=float)

        def is_walkable(r: int, c: int) -> bool:
            return grid[r, c] in (0, 2)

        dr, dc = d
        if not is_walkable(dr, dc):
            raise ValueError("Door must be on a walkable cell (0 or 2).")

        q = deque()
        dist[dr, dc] = 0.0
        q.append((dr, dc))

        while q:
            r, c = q.popleft()
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < rows and 0 <= cc < cols and is_walkable(rr, cc):
                    nd = dist[r, c] + 1.0
                    if dist[rr, cc] > nd:
                        dist[rr, cc] = nd
                        q.append((rr, cc))

        return dist

    def bfs_from(self, start: Cell) -> np.ndarray:
        """Compute BFS distance from a given walkable cell to every other walkable cell.

        Inputs: start cell.
        Returns: numpy array of distances, with infinity for unreachable cells.
        """
        grid = self.grid
        rows, cols = grid.shape
        dist = np.full((rows, cols), np.inf, dtype=float)
        sr, sc = start
        dist[sr, sc] = 0.0
        q = deque([(sr, sc)])
        while q:
            r, c = q.popleft()
            for rr, cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if 0 <= rr < rows and 0 <= cc < cols and grid[rr,cc] in (0,2):
                    if dist[rr,cc] == np.inf:
                        dist[rr,cc] = dist[r,c] + 1.0
                        q.append((rr,cc))
        return dist

    def bfs_distances_to_shelves(self, door: Optional[Cell] = None) -> List[Tuple[Cell, float]]:

        """Compute BFS distance from the door to each shelf cell.

        Inputs: optional door cell.
        Returns: list of pairs (cell, distance), sorted by distance ascending.
        """
        dist_corridor = self.corridor_distances(door)
        if self.grid is None:
            raise ValueError("Grid is not generated. Call generate() first.")
        grid = self.grid
        rows, cols = grid.shape

        shelves: List[Tuple[Cell, float]] = []
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 1:
                    continue
                neigh_dists = []
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < rows and 0 <= cc < cols and grid[rr, cc] in (0, 2):
                        neigh_dists.append(dist_corridor[rr, cc])
                if neigh_dists:
                    bfs_dist = float(min(neigh_dists)) + 1.0
                    shelves.append(((r, c), bfs_dist))

        shelves.sort(key=lambda x: x[1])
        return shelves

    def dijkstra_congestion(

        self,
        start: Cell,
        end: Cell,
        lam: float = 1.0,
    ) -> List[Cell]:
        """Shortest path with congestion-aware edge weights using Dijkstra.

        Inputs: start cell, end cell, lambda weight on edge congestion.
        Returns: list of cells from start to end inclusive, empty if no path.
        Edge cost is one plus lambda times the normalised edge congestion.
        Edges that were not traversed during the simulation use congestion zero.
        """
        if self.graph is None:
            self.build_graph()
        if self.edge_congestion is None:
            raise ValueError(
                "edge_congestion is not set. Call build_heatmap() first."
            )

        ec = self.edge_congestion

        def edge_cost(u: Cell, v: Cell) -> float:
            key = (u, v) if u <= v else (v, u)
            return 1.0 + lam * ec.get(key, 0.0)


        dist: Dict[Cell, float] = {start: 0.0}
        prev: Dict[Cell, Optional[Cell]] = {start: None}
        heap: List[Tuple[float, Cell]] = [(0.0, start)]

        while heap:
            cost, u = heapq.heappop(heap)
            if u == end:
                break
            if cost > dist.get(u, float("inf")):
                continue
            for v in self.graph.get(u, []):
                new_cost = cost + edge_cost(u, v)
                if new_cost < dist.get(v, float("inf")):
                    dist[v] = new_cost
                    prev[v] = u
                    heapq.heappush(heap, (new_cost, v))


        if end not in prev:
            return []
        path: List[Cell] = []
        cur: Optional[Cell] = end
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    def dijkstra_congestion_from(
        self,
        start: Cell,
        lam: float = 1.0,
    ) -> Tuple[Dict[Cell, float], Dict[Cell, Optional[Cell]]]:
        """Single-source Dijkstra with congestion-aware edge weights.

        Returns (dist, prev) dictionaries. Used by Dijkstra-NN visit ordering:
        one call per stop gives distances and predecessors to every cell,
        from which both the next nearest stop and its path are recovered.
        """
        if self.graph is None:
            self.build_graph()
        if self.edge_congestion is None:
            raise ValueError(
                "edge_congestion is not set. Call build_heatmap() first."
            )

        ec = self.edge_congestion

        def edge_cost(u: Cell, v: Cell) -> float:
            key = (u, v) if u <= v else (v, u)
            return 1.0 + lam * ec.get(key, 0.0)

        dist: Dict[Cell, float] = {start: 0.0}
        prev: Dict[Cell, Optional[Cell]] = {start: None}
        heap: List[Tuple[float, Cell]] = [(0.0, start)]

        while heap:
            cost, u = heapq.heappop(heap)
            if cost > dist.get(u, float("inf")):
                continue
            for v in self.graph.get(u, []):
                new_cost = cost + edge_cost(u, v)
                if new_cost < dist.get(v, float("inf")):
                    dist[v] = new_cost
                    prev[v] = u
                    heapq.heappush(heap, (new_cost, v))

        return dist, prev

    @staticmethod
    def _reconstruct_path(
        prev: Dict[Cell, Optional[Cell]],
        end: Cell,
    ) -> List[Cell]:
        """Reconstruct path from a Dijkstra predecessor map."""
        if end not in prev:
            return []
        path: List[Cell] = []
        cur: Optional[Cell] = end
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

