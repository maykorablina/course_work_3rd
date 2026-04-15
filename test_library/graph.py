from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple
from collections import deque
import numpy as np

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
