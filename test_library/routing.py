from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from .constants import Cell


class RoutingMixin:
    def generate_orders(
        self,
        num_orders: int = 10,
        items_per_order: int = 20,
        order_seed: Optional[int] = 42,
        weight_col: str = "Total_Annual_Units",
        skew: float = 2.0,
    ) -> List[List[str]]:
        """Generate a list of customer orders with realistic demand skew.

        Inputs: number of orders, items per order, optional seed, weight
        column name, skew exponent.
        Returns: list of orders; each order is a list of Item_ID strings.
        The probability of including an SKU is proportional to its weight
        raised to the power of skew. A higher skew makes A items more likely.
        """
        if self.abc_df is None:
            raise ValueError("Call abc_classify() first.")

        if self.shelf_assignments is not None:
            placed_ids: set = set()
            for cell_items in self.shelf_assignments.values():
                for item in cell_items:
                    placed_ids.add(item["Item_ID"])
            df_pool = self.abc_df[self.abc_df["Item_ID"].isin(placed_ids)].copy()
            print(f"[generate_orders] Pool: {len(df_pool)} placed SKUs "
                  f"(of {len(self.abc_df)} total, {len(self.abc_df) - len(df_pool)} not placed).")
        else:
            df_pool = self.abc_df.copy()

        items = df_pool["Item_ID"].values

        if weight_col not in df_pool.columns:
            weight_col = "Avg_Monthly_Demand"

        weights = df_pool[weight_col].values.astype(float)
        weights = np.clip(weights, 1e-9, None)
        if skew != 1.0:
            weights = weights ** skew
        probs = weights / weights.sum()

        if items_per_order > len(items):
            raise ValueError(
                f"items_per_order={items_per_order} > placed SKUs={len(items)}."
            )

        print(f"[generate_orders] {num_orders} orders × {items_per_order} items | "
              f"{weight_col}^{skew} | seed={order_seed}")

        orders = []
        rng = np.random.default_rng(seed=order_seed)
        for _ in range(num_orders):
            chosen = rng.choice(items, size=items_per_order, replace=False, p=probs)
            orders.append(chosen.tolist())

        return orders

    def _route_order_silent(self, order: List[str]) -> List[Cell]:
        """Route one order using BFS without drawing anything.

        Inputs: list of Item_IDs.
        Returns: list of cells forming the full route from door back to door.
        """
        if self.grid is None:
            raise ValueError("Call generate() first.")
        if self.shelf_assignments is None:
            raise ValueError("Call assign_items_to_shelves() first.")
        if self.graph is None:
            self.build_graph()

        category_rank = {
            "AX": 0, "AY": 1, "AZ": 2,
            "BX": 3, "BY": 4, "BZ": 5,
            "CX": 6, "CY": 7, "CZ": 8,
        }

        order_set = set(order)
        target_cells: Dict[Cell, dict] = {}

        for cell, items in self.shelf_assignments.items():
            for item in items:
                if item["Item_ID"] in order_set:
                    if cell not in target_cells:
                        target_cells[cell] = {"min_rank": 999, "min_dist": 999.0}
                    rank = category_rank.get(item["ABC_XYZ"], 999)
                    dist = item["BFS_Distance"]
                    if rank < target_cells[cell]["min_rank"]:
                        target_cells[cell]["min_rank"] = rank
                    if dist < target_cells[cell]["min_dist"]:
                        target_cells[cell]["min_dist"] = dist

        if not target_cells:
            return []

        dist_corridor = self.corridor_distances()

        def get_access_cell(shelf_cell: Cell) -> Optional[Cell]:
            r, c = shelf_cell
            grid_rows, grid_cols = self.grid.shape
            best_neighbor = None
            min_d = float("inf")
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < grid_rows and 0 <= cc < grid_cols and self.grid[rr, cc] in (0, 2):
                    if dist_corridor[rr, cc] < min_d:
                        min_d = dist_corridor[rr, cc]
                        best_neighbor = (rr, cc)
            return best_neighbor

        access_pairs = []
        for shelf, info in target_cells.items():
            ac = get_access_cell(shelf)
            if ac is not None:
                access_pairs.append((ac, info["min_dist"]))

        ordered_access = [ac for ac, _ in sorted(access_pairs, key=lambda p: p[1])]

        full_path: List[Cell] = []
        current_pos = self.door

        for access_cell in ordered_access:
            segment = self.shortest_path_bfs(current_pos, access_cell)
            if segment:
                if full_path and full_path[-1] == segment[0]:
                    full_path.extend(segment[1:])
                else:
                    full_path.extend(segment)
                current_pos = access_cell

        return_seg = self.shortest_path_bfs(current_pos, self.door)
        if return_seg:
            if full_path and full_path[-1] == return_seg[0]:
                full_path.extend(return_seg[1:])
            else:
                full_path.extend(return_seg)

        return full_path

    def batch_route_lengths(
        self,
        orders: List[List[str]],
    ) -> List[int]:
        """Compute the route length for every order using BFS routing.

        Inputs: list of orders.
        Returns: list of integer route lengths in steps.
        """
        lengths: List[int] = []
        for order in orders:
            path = self._route_order_silent(order)
            lengths.append(len(path))
        return lengths
