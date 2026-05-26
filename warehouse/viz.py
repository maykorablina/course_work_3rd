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


class VizMixin:
    def draw_grid(
        self,
        title: str = "Warehouse grid map",
        show_door: bool = True,
        figsize_scale: float = 2.2,
        save_path: Optional[str] = None,
    ) -> None:
        """Draw the warehouse grid as a coloured image.

        Inputs: title, show_door flag, figsize_scale, optional save_path.
        Returns: None. Shows the figure and optionally writes it to disk.
        The door cell is marked with the letter D.
        """
        if self.grid is None:
            raise ValueError("Grid is not generated. Call generate() first.")

        grid = self.grid
        cmap = ListedColormap([
            "#2b2b2b",
            "#f28c28",
            "#3d6df2",
        ])

        rows, cols = grid.shape
        fig, ax = plt.subplots(figsize=(cols / figsize_scale, rows / figsize_scale))
        ax.imshow(grid, cmap=cmap)

        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="#6a6a6a", linewidth=0.8)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

        if show_door:
            dr, dc = self.door
            ax.text(dc, dr, "D", ha="center", va="center",
                    fontsize=14, color="white", fontweight="bold")

        plt.title(title)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.show()

    def draw_abc_zones(
        self,
        title: str = "Warehouse ABC-XYZ zones",
        figsize_scale: float = 2.2,
    ) -> None:
        """Draw the warehouse grid with shelves coloured by ABC_XYZ category.

        Inputs: title, figsize_scale.
        Returns: None. Shows the figure. AX uses green, CZ uses red, and
        intermediate categories form a smooth green to yellow to red gradient.
        """
        if self.grid is None:
            raise ValueError("Call generate() first.")
        if self.shelf_assignments is None:
            raise ValueError("Call assign_items_to_shelves() first.")


        category_rank = {
            "AX": 0, "AY": 1, "AZ": 2,
            "BX": 3, "BY": 4, "BZ": 5,
            "CX": 6, "CY": 7, "CZ": 8,
        }

        def rank_to_color(rank: int) -> np.ndarray:
            """Draw the warehouse grid with shelves coloured by ABC_XYZ category.

            Inputs: title, figsize_scale.
            Returns: None. Shows the figure.
            """
            t = rank / 8.0
            if t <= 0.5:

                s = t / 0.5
                return np.array([s, 0.80, 0.30 * (1 - s)])
            else:

                s = (t - 0.5) / 0.5
                return np.array([1.0, 0.80 * (1 - s), 0.0])

        grid = self.grid
        rows, cols = grid.shape

        rgb = np.zeros((rows, cols, 3))
        color_passage = np.array([0.17, 0.17, 0.17])
        color_shelf   = np.array([0.55, 0.55, 0.55])
        color_door    = np.array([0.24, 0.43, 0.95])

        for r in range(rows):
            for c in range(cols):
                v = grid[r, c]
                if v == 0:
                    rgb[r, c] = color_passage
                elif v == 2:
                    rgb[r, c] = color_door
                elif v == 1:
                    items_here = self.shelf_assignments.get((r, c), [])
                    if items_here:
                        cat = items_here[0]["ABC_XYZ"]
                        rank = category_rank.get(cat, 4)
                        rgb[r, c] = rank_to_color(rank)
                    else:
                        rgb[r, c] = color_shelf

        fig, ax = plt.subplots(figsize=(cols / figsize_scale, rows / figsize_scale))
        ax.imshow(rgb)

        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="#6a6a6a", linewidth=0.8)
        ax.tick_params(which="both", bottom=False, left=False,
                       labelbottom=False, labelleft=False)


        dr, dc = self.door
        ax.text(dc, dr, "D", ha="center", va="center",
                fontsize=14, color="white", fontweight="bold")


        legend_elements = []
        for cat in ["AX", "AY", "AZ", "BX", "BY", "BZ", "CX", "CY", "CZ"]:
            legend_elements.append(
                Patch(facecolor=rank_to_color(category_rank[cat]), label=cat)
            )
        legend_elements.append(Patch(facecolor=color_door, label="Door"))
        ax.legend(handles=legend_elements, loc="upper right", fontsize=7, ncol=2)

    def process_order(
        self,
        order: List[str],
        title: str = "Order Routing",
        figsize_scale: float = 2.2,
        save_path: Optional[str] = None,
    ) -> Tuple[List[Cell], pd.DataFrame]:
        """Route one order using BFS and draw the resulting path.

        Inputs: list of Item_IDs, title, figsize_scale, optional save_path.
        Returns: tuple (full path as list of cells, DataFrame of items in
        visit order). The route starts and ends at the door.
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
        target_cells = {}

        for cell, items in self.shelf_assignments.items():
            for item in items:
                if item["Item_ID"] in order_set:
                    if cell not in target_cells:
                        target_cells[cell] = {
                            "min_rank": 999,
                            "min_dist": 999.0,
                            "items": []
                        }
                    rank = category_rank.get(item["ABC_XYZ"], 999)
                    dist = item["BFS_Distance"]
                    if rank < target_cells[cell]["min_rank"]:
                        target_cells[cell]["min_rank"] = rank
                    if dist < target_cells[cell]["min_dist"]:
                        target_cells[cell]["min_dist"] = dist

                    target_cells[cell]["items"].append(
                        (item["Item_ID"], item["ABC_XYZ"], item["Total_Sales_Value"])
                    )

        if not target_cells:
            print("No items from the order were found on the shelves.")
            return [], pd.DataFrame()


        dist_corridor = self.corridor_distances()

        def get_access_cell(shelf_cell: Cell) -> Optional[Cell]:
            r, c = shelf_cell
            rows_, cols_ = self.grid.shape
            best_neighbor = None
            min_d = float('inf')
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < rows_ and 0 <= cc < cols_ and self.grid[rr, cc] in (0, 2):
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


        full_path = []
        current_pos = self.door
        stops = [self.door]

        for access_cell in ordered_access:
            path_segment = self.shortest_path_bfs(current_pos, access_cell)
            if path_segment:
                if full_path and full_path[-1] == path_segment[0]:
                    full_path.extend(path_segment[1:])
                else:
                    full_path.extend(path_segment)
                current_pos = access_cell
                stops.append(access_cell)


        return_path = self.shortest_path_bfs(current_pos, self.door)
        if return_path:
            if full_path and full_path[-1] == return_path[0]:
                full_path.extend(return_path[1:])
            else:
                full_path.extend(return_path)
            stops.append(self.door)



        def rank_to_color(rank: int) -> np.ndarray:
            t = rank / 8.0
            if t <= 0.5:
                s = t / 0.5
                return np.array([s, 0.80, 0.30 * (1 - s)])
            else:
                s = (t - 0.5) / 0.5
                return np.array([1.0, 0.80 * (1 - s), 0.0])

        rows, cols = self.grid.shape
        rgb = np.zeros((rows, cols, 3))
        color_passage = np.array([0.17, 0.17, 0.17])
        color_shelf   = np.array([0.55, 0.55, 0.55])
        color_door    = np.array([0.24, 0.43, 0.95])

        for r in range(rows):
            for c in range(cols):
                v = self.grid[r, c]
                if v == 0:
                    rgb[r, c] = color_passage
                elif v == 2:
                    rgb[r, c] = color_door
                elif v == 1:
                    items_here = self.shelf_assignments.get((r, c), [])
                    if items_here:
                        cat = items_here[0]["ABC_XYZ"]
                        rank = category_rank.get(cat, 4)
                        rgb[r, c] = rank_to_color(rank)
                    else:
                        rgb[r, c] = color_shelf

        fig, ax = plt.subplots(figsize=(cols / figsize_scale, rows / figsize_scale))
        ax.imshow(rgb)

        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="#6a6a6a", linewidth=0.8)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

        dr, dc = self.door
        ax.text(dc, dr, "D", ha="center", va="center", fontsize=18, color="white", fontweight="bold", zorder=10)


        if full_path:
            path_y = [p[0] for p in full_path]
            path_x = [p[1] for p in full_path]
            ax.plot(path_x, path_y, color="#ff00ff", linewidth=4.0, alpha=0.7, label="Route")


            arrow_interval = 4
            for i in range(0, len(path_x) - 1, arrow_interval):
                dx = path_x[i+1] - path_x[i]
                dy = path_y[i+1] - path_y[i]
                if dx != 0 or dy != 0:
                    ax.annotate('', xy=(path_x[i+1], path_y[i+1]), xytext=(path_x[i], path_y[i]),
                                arrowprops=dict(arrowstyle="-|>,head_width=0.8,head_length=1.2",
                                                color="#ff00ff", lw=3.0), zorder=4)


            stop_y = [p[0] for p in stops]
            stop_x = [p[1] for p in stops]
            ax.scatter(stop_x, stop_y, color="white", s=150, zorder=5)


            for i, p in enumerate(stops[1:-1], 1):
                ax.text(p[1], p[0] - 0.4, str(i), color="white", fontsize=20, ha="center", va="center", fontweight="bold",
                       bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'), zorder=6)


        legend_elements = [
            Patch(facecolor=color_door, label="Door"),
            plt.Line2D([0], [0], color="#ff00ff", lw=2.5, label=f"Path length: {len(full_path)} steps")
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

        plt.title(title)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.show()


        order_info = []
        for cell, data in target_cells.items():
            for item_id, cat, val in data["items"]:
                order_info.append({
                    "Item_ID": item_id,
                    "ABC_XYZ": cat,
                    "Total_Sales_Value": val,
                    "Shelf_Cell": cell
                })

        df_order = pd.DataFrame(order_info).sort_values(by=["ABC_XYZ", "Total_Sales_Value"], ascending=[True, False]).reset_index(drop=True)

        return full_path, df_order

    def draw_heatmap(
        self,
        title: str = "Warehouse Traffic Heatmap",
        figsize_scale: float = 2.2,
    ) -> None:
        """Plot the traffic heatmap over the warehouse layout.

        Inputs: title, figsize_scale.
        Returns: None. Shows the figure. Shelves are grey, the door is blue,
        and walkable cells use a yellow to red colour scale by visit count.
        """
        if self.heatmap is None:
            raise ValueError("Call build_heatmap() first.")

        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize, PowerNorm

        grid = self.grid
        grid_rows, grid_cols = grid.shape




        if self.cell_counts is not None:
            display = self.cell_counts.astype(float)
        else:
            display = self.heatmap.astype(float)

        vmax = float(display.max()) if display.max() > 0 else 1.0


        bg = np.zeros((grid_rows, grid_cols, 3), dtype=float)
        for r in range(grid_rows):
            for c in range(grid_cols):
                v = grid[r, c]
                if v == 1:
                    bg[r, c] = [0.45, 0.45, 0.45]
                elif v == 2:
                    bg[r, c] = [0.24, 0.43, 0.95]
                else:
                    bg[r, c] = [0.08, 0.08, 0.12]

        fig, ax = plt.subplots(
            figsize=(grid_cols / figsize_scale, grid_rows / figsize_scale)
        )
        ax.imshow(bg, interpolation="nearest")


        gamma = 0.4
        norm = PowerNorm(gamma=gamma, vmin=0, vmax=vmax)

        walkable_mask = (grid == 0)
        data_masked = np.ma.masked_where(~walkable_mask, display)
        im = ax.imshow(
            data_masked,
            cmap="YlOrRd",
            norm=norm,
            alpha=0.92,
            interpolation="nearest",
        )


        ax.set_xticks(np.arange(-0.5, grid_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_rows, 1), minor=True)
        ax.grid(which="minor", color="#555555", linewidth=0.5)
        ax.tick_params(
            which="both", bottom=False, left=False,
            labelbottom=False, labelleft=False,
        )


        dr, dc = self.door
        ax.text(
            dc, dr, "D",
            ha="center", va="center",
            fontsize=14, color="white", fontweight="bold",
        )


        cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
        cbar.set_label(f"Cell visits (max = {int(vmax):,})", fontsize=9)


        if self.cell_counts is not None:
            total_visits = int(self.cell_counts.sum())
            hot_cells = int((display > vmax * 0.75).sum())
            ax.set_xlabel(
                f"Total cell visits: {total_visits:,}   |   "
                f"Max: {int(vmax):,} visits   |   "
                f"Hot cells (>75% max): {hot_cells}",
                fontsize=8,
            )

        plt.title(title, fontsize=11)
        plt.tight_layout()
        plt.show()

    def process_order_congestion_aware(
        self,
        order: List[str],
        lam: float = 1.0,
        title: str = "Congestion-Aware Routing",
        figsize_scale: float = 2.2,
        save_path: Optional[str] = None,
    ) -> Tuple[List[Cell], pd.DataFrame]:
        """Route one order with congestion-aware Dijkstra and draw the path.

        Inputs: list of Item_IDs, lambda weight, title, figsize_scale,
        optional save_path.
        Returns: tuple (full path as list of cells, DataFrame of items).
        """
        if self.grid is None:
            raise ValueError("Call generate() first.")
        if self.shelf_assignments is None:
            raise ValueError("Call assign_items_to_shelves() first.")
        if self.edge_congestion is None:
            raise ValueError("Call build_heatmap() first.")
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
                        target_cells[cell] = {
                            "min_rank": 999, "min_dist": 999.0, "items": []
                        }
                    rank = category_rank.get(item["ABC_XYZ"], 999)
                    dist_val = item["BFS_Distance"]
                    if rank < target_cells[cell]["min_rank"]:
                        target_cells[cell]["min_rank"] = rank
                    if dist_val < target_cells[cell]["min_dist"]:
                        target_cells[cell]["min_dist"] = dist_val
                    target_cells[cell]["items"].append(
                        (item["Item_ID"], item["ABC_XYZ"], item["Total_Sales_Value"])
                    )

        if not target_cells:
            print("No items from the order were found on the shelves.")
            return [], pd.DataFrame()

        dist_corridor = self.corridor_distances()

        def get_access_cell(shelf_cell: Cell) -> Optional[Cell]:
            r, c = shelf_cell
            rows, cols = self.grid.shape
            best = None
            min_d = float("inf")
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < rows and 0 <= cc < cols and self.grid[rr, cc] in (0, 2):
                    if dist_corridor[rr, cc] < min_d:
                        min_d = dist_corridor[rr, cc]
                        best = (rr, cc)
            return best

        access_for: Dict[Cell, Cell] = {}
        for shelf in target_cells:
            ac = get_access_cell(shelf)
            if ac is not None:
                access_for[shelf] = ac

        remaining = list(access_for.values())
        full_path: List[Cell] = []
        current_pos = self.door
        stops = [self.door]

        while remaining:
            dist_map, prev_map = self.dijkstra_congestion_from(current_pos, lam=lam)
            nearest = min(remaining, key=lambda c: dist_map.get(c, float("inf")))
            segment = self._reconstruct_path(prev_map, nearest)
            if not segment:
                remaining.remove(nearest)
                continue
            if full_path and full_path[-1] == segment[0]:
                full_path.extend(segment[1:])
            else:
                full_path.extend(segment)
            current_pos = nearest
            stops.append(nearest)
            remaining.remove(nearest)

        dist_map, prev_map = self.dijkstra_congestion_from(current_pos, lam=lam)
        return_seg = self._reconstruct_path(prev_map, self.door)
        if return_seg:
            if full_path and full_path[-1] == return_seg[0]:
                full_path.extend(return_seg[1:])
            else:
                full_path.extend(return_seg)
            stops.append(self.door)


        def rank_to_color(rank: int) -> np.ndarray:
            t = rank / 8.0
            if t <= 0.5:
                s = t / 0.5
                return np.array([s, 0.80, 0.30 * (1 - s)])
            else:
                s = (t - 0.5) / 0.5
                return np.array([1.0, 0.80 * (1 - s), 0.0])

        rows, cols = self.grid.shape
        rgb = np.zeros((rows, cols, 3))
        color_passage = np.array([0.17, 0.17, 0.17])
        color_shelf   = np.array([0.55, 0.55, 0.55])
        color_door    = np.array([0.24, 0.43, 0.95])

        for r in range(rows):
            for c in range(cols):
                v = self.grid[r, c]
                if v == 0:
                    rgb[r, c] = color_passage
                elif v == 2:
                    rgb[r, c] = color_door
                elif v == 1:
                    items_here = self.shelf_assignments.get((r, c), [])
                    if items_here:
                        cat = items_here[0]["ABC_XYZ"]
                        rank = category_rank.get(cat, 4)
                        rgb[r, c] = rank_to_color(rank)
                    else:
                        rgb[r, c] = color_shelf

        fig, ax = plt.subplots(figsize=(cols / figsize_scale, rows / figsize_scale))
        ax.imshow(rgb)

        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="#6a6a6a", linewidth=0.8)
        ax.tick_params(which="both", bottom=False, left=False,
                       labelbottom=False, labelleft=False)

        dr, dc = self.door
        ax.text(dc, dr, "D", ha="center", va="center",
                fontsize=18, color="white", fontweight="bold", zorder=10)

        if full_path:
            path_y = [p[0] for p in full_path]
            path_x = [p[1] for p in full_path]
            ax.plot(path_x, path_y, color="#00e5ff", linewidth=4.0,
                    alpha=0.75, label="Congestion-aware route")

            arrow_interval = 4
            for i in range(0, len(path_x) - 1, arrow_interval):
                dx = path_x[i + 1] - path_x[i]
                dy = path_y[i + 1] - path_y[i]
                if dx != 0 or dy != 0:
                    ax.annotate(
                        "",
                        xy=(path_x[i + 1], path_y[i + 1]),
                        xytext=(path_x[i], path_y[i]),
                        arrowprops=dict(
                            arrowstyle="-|>,head_width=0.8,head_length=1.2",
                            color="#00e5ff", lw=3.0,
                        ),
                        zorder=4,
                    )

            stop_y = [p[0] for p in stops]
            stop_x = [p[1] for p in stops]
            ax.scatter(stop_x, stop_y, color="white", s=150, zorder=5)

            for i, p in enumerate(stops[1:-1], 1):
                ax.text(
                    p[1], p[0] - 0.4, str(i),
                    color="white", fontsize=20, ha="center", va="center",
                    fontweight="bold",
                    bbox=dict(facecolor="black", alpha=0.7, boxstyle="round,pad=0.3"),
                    zorder=6,
                )

        legend_elements = [
            Patch(facecolor=color_door, label="Door"),
            plt.Line2D([0], [0], color="#00e5ff", lw=2.5,
                       label=f"Path length: {len(full_path)} steps  (λ={lam})"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

        plt.title(title)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.show()

        order_info = []
        for cell, data in target_cells.items():
            for item_id, cat, val in data["items"]:
                order_info.append({
                    "Item_ID": item_id, "ABC_XYZ": cat,
                    "Total_Sales_Value": val, "Shelf_Cell": cell,
                })
        df_order = (
            pd.DataFrame(order_info)
            .sort_values(by=["ABC_XYZ", "Total_Sales_Value"],
                         ascending=[True, False])
            .reset_index(drop=True)
        )
        return full_path, df_order
