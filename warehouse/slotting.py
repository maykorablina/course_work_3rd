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


class SlottingMixin:
    def abc_classify(
        self,
        csv_path: str,
        a_pct: float = 0.20,
        b_pct: float = 0.30,
        x_cv: float = 0.10,
        y_cv: float = 0.25,
    ) -> pd.DataFrame:
        """Load the SKU table and assign ABC and XYZ categories.

        Inputs: csv path, A and B share thresholds, X and Y CV thresholds.
        Returns: DataFrame with added columns ABC_Category, XYZ_Category,
        ABC_XYZ, Avg_Monthly_Demand, Std_Monthly_Demand, CV, Shelves_Needed.
        ABC uses the APICS share split. XYZ uses the standard CV thresholds.
        """
        df = pd.read_csv(csv_path)
        n = len(df)


        df = df.sort_values("Total_Annual_Units", ascending=False).reset_index(drop=True)

        a_count = int(np.ceil(n * a_pct))
        b_count = int(np.ceil(n * b_pct))
        c_count = n - a_count - b_count

        abc_labels = ["A"] * a_count + ["B"] * b_count + ["C"] * c_count
        df["ABC_Category"] = abc_labels[:n]


        month_cols = [c for c in df.columns if c.endswith("_Demand")]
        monthly = df[month_cols]
        df["Avg_Monthly_Demand"] = monthly.mean(axis=1).round(1)
        df["Std_Monthly_Demand"] = monthly.std(axis=1, ddof=1)
        df["CV"] = (df["Std_Monthly_Demand"] / df["Avg_Monthly_Demand"]).round(4)

        xyz_cond = [
            df["CV"] <= x_cv,
            df["CV"] <= y_cv,
        ]
        df["XYZ_Category"] = np.select(xyz_cond, ["X", "Y"], default="Z")


        df["ABC_XYZ"] = df["ABC_Category"] + df["XYZ_Category"]


        df["Shelves_Needed"] = np.ceil(
            df["Avg_Monthly_Demand"] / max(self.shelf_capacity, 1)
        ).astype(int)

        self.abc_df = df
        return df

    def summary_table(self) -> pd.DataFrame:
        """Return a short summary of the ABC and XYZ classification.

        Inputs: none.
        Returns: DataFrame with columns Item_ID, ABC_XYZ, Avg_Monthly_Demand, Total_Sales_Value, Shelves_Needed.
        """
        if self.abc_df is None:
            raise ValueError("Call abc_classify() first.")
        return self.abc_df[["Item_ID", "ABC_XYZ", "Avg_Monthly_Demand", "Total_Sales_Value", "Shelves_Needed"]].copy()

    def assign_items_to_shelves(self) -> Dict[Cell, List[dict]]:
        """Place items on shelves in order of ABC_XYZ rank and sales value.

        Inputs: none, uses self.abc_df and self.grid.
        Returns: mapping shelf_cell to list of item records. Items in the
        top ranks go to shelves closer to the door.
        """
        if self.abc_df is None:
            raise ValueError("Call abc_classify() first.")
        if self.grid is None:
            raise ValueError("Call generate() first.")


        shelves = self.bfs_distances_to_shelves()
        self.shelf_cells = shelves


        rank_order = {"AX": 0, "AY": 1, "AZ": 2, "BX": 3, "BY": 4, "BZ": 5,
                      "CX": 6, "CY": 7, "CZ": 8}
        items = self.abc_df.copy()
        items["_rank"] = items["ABC_XYZ"].map(rank_order)
        items = items.sort_values(
            by=["_rank", "Total_Annual_Units"],
            ascending=[True, False],
        ).reset_index(drop=True)
        items = items.drop(columns=["_rank"])

        assignments: Dict[Cell, List[dict]] = {}
        shelf_idx = 0
        total_shelves = len(shelves)

        for _, row in items.iterrows():
            needed = 1
            for _ in range(needed):
                if shelf_idx >= total_shelves:
                    break
                cell, dist = shelves[shelf_idx]
                if cell not in assignments:
                    assignments[cell] = []
                assignments[cell].append({
                    "Item_ID": row["Item_ID"],
                    "Item_Name": row["Item_Name"],
                    "ABC_Category": row["ABC_Category"],
                    "ABC_XYZ": row["ABC_XYZ"],
                    "Total_Annual_Units": row["Total_Annual_Units"],
                    "Total_Sales_Value": row["Total_Sales_Value"],
                    "BFS_Distance": dist,
                })
                shelf_idx += 1

        self.shelf_assignments = assignments
        return assignments

    def _shelf_access_congestion(self) -> Dict[Cell, float]:
        """Estimate congestion at each shelf based on its access cell.

        Inputs: none, uses the current heatmap.
        Returns: mapping shelf cell to congestion value in zero to one.
        """
        if self.heatmap is None:
            raise ValueError("Call build_heatmap() first.")

        dist_corridor = self.corridor_distances()
        grid = self.grid
        rows_, cols_ = grid.shape

        def get_access(shelf_cell: Cell) -> Optional[Cell]:
            r, c = shelf_cell
            best, min_d = None, float('inf')
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= rr < rows_ and 0 <= cc < cols_ and grid[rr, cc] in (0, 2):
                    if dist_corridor[rr, cc] < min_d:
                        min_d = dist_corridor[rr, cc]
                        best = (rr, cc)
            return best

        result: Dict[Cell, float] = {}
        for cell in self.shelf_assignments:
            ac = get_access(cell)
            result[cell] = float(self.heatmap[ac[0], ac[1]]) if ac is not None else 0.0
        return result

    def heatmap_slotting(
        self,
        orders: List[List[str]],
        hot_pct: float = 0.20,
        cold_pct_start: float = 0.40,
        max_swaps: Optional[int] = None,
        rebuild: bool = True,
        verbose: bool = True,
    ) -> dict:
        """Swap A items on hot shelves with C items on cold shelves.

        Inputs: orders, hot share, cold share start, optional swap cap,
        rebuild flag, verbose flag.
        Returns: dict with Gini before and after, number of swaps, and
        list of swapped pairs.
        """
        if self.shelf_assignments is None:
            raise ValueError("Call assign_items_to_shelves() first.")
        if self.heatmap is None:
            if verbose:
                print("[heatmap_slotting] No heatmap available, building it...")
            self.build_heatmap(orders)


        gini_before = self._gini_coefficient(self.cell_counts)


        shelf_cong = self._shelf_access_congestion()
        dist_corr  = self.corridor_distances()
        grid = self.grid
        rows_, cols_ = grid.shape

        def bfs_dist_to_shelf(cell: Cell) -> float:
            """Swap A items on hot shelves with C items on cold shelves.

            Inputs: orders, hot share, cold share start, optional swap cap, rebuild flag, verbose flag.
            Returns: dict with Gini before and after, number of swaps, and list of swapped pairs.
            """
            r, c = cell
            best = float('inf')
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= rr < rows_ and 0 <= cc < cols_ and grid[rr, cc] in (0, 2):
                    best = min(best, dist_corr[rr, cc])
            return best


        sorted_cells = sorted(shelf_cong.keys(), key=lambda c: -shelf_cong[c])
        n_total = len(sorted_cells)
        n_hot   = max(1, int(n_total * hot_pct))
        n_cold_start = int(n_total * cold_pct_start)

        hot_set  = set(sorted_cells[:n_hot])
        cold_set = set(sorted_cells[n_cold_start:])



        hot_candidates = sorted(
            [c for c in hot_set
             if any(item["ABC_XYZ"][0] == "A" for item in self.shelf_assignments[c])],
            key=lambda c: -shelf_cong[c]
        )



        cold_candidates = sorted(
            [c for c in cold_set
             if any(item["ABC_XYZ"][0] == "C" for item in self.shelf_assignments[c])],
            key=lambda c: (shelf_cong[c], bfs_dist_to_shelf(c))
        )

        n_swaps_possible = min(len(hot_candidates), len(cold_candidates))
        if max_swaps is not None:
            n_swaps_possible = min(n_swaps_possible, max_swaps)

        if verbose:
            print(f"[heatmap_slotting] Hot shelves: {len(hot_set)}, "
                  f"Cold shelves: {len(cold_set)}")
            print(f"[heatmap_slotting] Hot A-candidates: {len(hot_candidates)}, "
                  f"Cold C-candidates: {len(cold_candidates)}")
            print(f"[heatmap_slotting] Will perform {n_swaps_possible} swaps")


        swapped_pairs = []
        for i in range(n_swaps_possible):
            hot_cell  = hot_candidates[i]
            cold_cell = cold_candidates[i]


            self.shelf_assignments[hot_cell], self.shelf_assignments[cold_cell] = \
                self.shelf_assignments[cold_cell], self.shelf_assignments[hot_cell]


            new_dist_hot  = float(bfs_dist_to_shelf(hot_cell))
            new_dist_cold = float(bfs_dist_to_shelf(cold_cell))
            for item in self.shelf_assignments[hot_cell]:
                item["BFS_Distance"] = new_dist_hot
            for item in self.shelf_assignments[cold_cell]:
                item["BFS_Distance"] = new_dist_cold

            swapped_pairs.append((hot_cell, cold_cell))


        if rebuild:
            if verbose:
                print("[heatmap_slotting] Rebuilding heatmap after slotting...")
            self.build_heatmap(orders)
            gini_after = self._gini_coefficient(self.cell_counts)
        else:
            gini_after = None

        result = {
            "gini_before":      gini_before,
            "gini_after":       gini_after,
            "gini_delta":       (gini_after - gini_before) if gini_after is not None else None,
            "n_swaps":          n_swaps_possible,
            "hot_shelves":      len(hot_set),
            "cold_shelves":     len(cold_set),
            "hot_candidates":   len(hot_candidates),
            "cold_candidates":  len(cold_candidates),
            "swapped_pairs":    swapped_pairs,
        }

        if verbose:
            print(f"[heatmap_slotting] Gini BEFORE: {gini_before:.4f}")
            if gini_after is not None:
                delta = gini_after - gini_before
                print(f"[heatmap_slotting] Gini AFTER:  {gini_after:.4f}  "
                      f"(Δ = {delta:+.4f}, {delta/gini_before*100:+.1f}%)")

        return result

    def dwc_slotting(
        self,
        orders: List[List[str]],
        dist_tol: float = 5.0,
        max_swaps: Optional[int] = None,
        rebuild: bool = True,
        verbose: bool = True,
    ) -> dict:
        """Demand-Weighted Congestion slotting that limits walking distance.

        Inputs: orders, distance tolerance, optional swap cap, rebuild flag,
        verbose flag.
        Returns: dict with Gini before and after, number of swaps, and list
        of swapped pairs. A swap is accepted only if the BFS distance gap
        is within the tolerance and the new congestion is lower.
        """
        if self.shelf_assignments is None:
            raise ValueError("Call assign_items_to_shelves() first.")
        if self.heatmap is None:
            if verbose:
                print("[dwc_slotting] No heatmap available, building it...")
            self.build_heatmap(orders)

        gini_before = self._gini_coefficient(self.cell_counts)


        shelf_cong = self._shelf_access_congestion()
        dist_corr  = self.corridor_distances()
        grid       = self.grid
        rows_, cols_ = grid.shape

        def bfs_dist_to_shelf(cell: Cell) -> float:
            r, c = cell
            best = float('inf')
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= rr < rows_ and 0 <= cc < cols_ and grid[rr, cc] in (0, 2):
                    best = min(best, dist_corr[rr, cc])
            return best


        shelf_dist = {cell: bfs_dist_to_shelf(cell) for cell in self.shelf_assignments}


        def shelf_demand(cell: Cell) -> float:
            return sum(
                item.get("Total_Annual_Units", 1.0)
                for item in self.shelf_assignments[cell]
            )


        a_shelves = sorted(
            [c for c in self.shelf_assignments
             if any(item["ABC_XYZ"][0] == "A" for item in self.shelf_assignments[c])],
            key=lambda c: -shelf_demand(c) * shelf_cong[c]
        )


        c_shelves = sorted(
            [c for c in self.shelf_assignments
             if any(item["ABC_XYZ"][0] == "C" for item in self.shelf_assignments[c])],
            key=lambda c: shelf_cong[c]
        )

        if verbose:
            print(f"[dwc_slotting] A-shelves: {len(a_shelves)}, C-shelves: {len(c_shelves)}")
            print(f"[dwc_slotting] Distance tolerance: ±{dist_tol} steps")

        used_c = set()
        swapped_pairs = []
        n_done = 0

        for a_cell in a_shelves:
            if max_swaps is not None and n_done >= max_swaps:
                break
            a_dist = shelf_dist[a_cell]
            a_cong = shelf_cong[a_cell]


            best_c = None
            best_cong = a_cong

            for c_cell in c_shelves:
                if c_cell in used_c:
                    continue
                c_dist = shelf_dist[c_cell]
                if abs(c_dist - a_dist) > dist_tol:
                    continue
                c_cong = shelf_cong[c_cell]
                if c_cong < best_cong:
                    best_cong = c_cong
                    best_c = c_cell

            if best_c is None:
                continue


            self.shelf_assignments[a_cell], self.shelf_assignments[best_c] = \
                self.shelf_assignments[best_c], self.shelf_assignments[a_cell]

            new_dist_a = float(shelf_dist[a_cell])
            new_dist_c = float(shelf_dist[best_c])
            for item in self.shelf_assignments[a_cell]:
                item["BFS_Distance"] = new_dist_a
            for item in self.shelf_assignments[best_c]:
                item["BFS_Distance"] = new_dist_c

            used_c.add(best_c)
            swapped_pairs.append((a_cell, best_c,
                                   round(a_cong, 4), round(best_cong, 4)))
            n_done += 1

        if verbose:
            print(f"[dwc_slotting] Swaps performed: {n_done}")

        if rebuild:
            if verbose:
                print("[dwc_slotting] Rebuilding heatmap...")
            self.build_heatmap(orders)
            gini_after = self._gini_coefficient(self.cell_counts)
        else:
            gini_after = None

        result = {
            "gini_before":   gini_before,
            "gini_after":    gini_after,
            "gini_delta":    (gini_after - gini_before) if gini_after is not None else None,
            "n_swaps":       n_done,
            "swapped_pairs": swapped_pairs,
        }

        if verbose:
            print(f"[dwc_slotting] Gini BEFORE: {gini_before:.4f}")
            if gini_after is not None:
                delta = gini_after - gini_before
                print(f"[dwc_slotting] Gini AFTER:  {gini_after:.4f}  "
                      f"(Δ = {delta:+.4f}, {delta/gini_before*100:+.1f}%)")

        return result

    def taps_slotting(
        self,
        orders: List[List[str]],
        alpha: float = 3.0,
        rebuild: bool = True,
        verbose: bool = True,
    ) -> dict:
        """Traffic-Aware Prioritised Slotting with full reassignment.

        Inputs: orders, alpha weight on path congestion, rebuild flag,
        verbose flag.
        Returns: dict with Gini before and after, alpha, item and shelf
        counts. Higher alpha avoids hot corridors even at the cost of
        longer paths.
        """
        if self.shelf_assignments is None:
            raise ValueError("Call assign_items_to_shelves() first.")
        if self.heatmap is None:
            if verbose:
                print("[taps_slotting] No heatmap available, building it...")
            self.build_heatmap(orders)

        gini_before = self._gini_coefficient(self.cell_counts)

        dist_corr   = self.corridor_distances()
        grid        = self.grid
        rows_, cols_ = grid.shape


        def get_access(cell: Cell) -> Optional[Cell]:
            r, c = cell
            best, min_d = None, float('inf')
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= rr < rows_ and 0 <= cc < cols_ and grid[rr, cc] in (0, 2):
                    if dist_corr[rr, cc] < min_d:
                        min_d = dist_corr[rr, cc]
                        best = (rr, cc)
            return best


        def approach_wpc(cell: Cell) -> float:
            ac = get_access(cell)
            if ac is None:
                return float('inf')
            path = self.shortest_path_bfs(self.door, ac)
            return sum(float(self.heatmap[r, c]) for r, c in path)


        def bfs_d(cell: Cell) -> float:
            ac = get_access(cell)
            return float(dist_corr[ac[0], ac[1]]) if ac is not None else float('inf')


        if verbose:
            print("[taps_slotting] Computing approach scores for all shelves...")
        shelf_score: Dict[Cell, float] = {}
        shelf_dist_cache: Dict[Cell, float] = {}
        for cell in self.shelf_assignments:
            d = bfs_d(cell)
            w = approach_wpc(cell)
            shelf_score[cell]      = d + alpha * w
            shelf_dist_cache[cell] = d


        all_items: list = []
        for items in self.shelf_assignments.values():
            all_items.extend(items)


        all_items.sort(key=lambda x: x.get("Total_Annual_Units", 0.0), reverse=True)


        sorted_shelves = sorted(
            self.shelf_assignments.keys(),
            key=lambda c: shelf_score[c]
        )

        if verbose:
            top5 = [(c, round(shelf_score[c], 2), round(shelf_dist_cache[c], 1))
                    for c in sorted_shelves[:5]]
            print(f"[taps_slotting] Top-5 best shelves (cell, score, dist): {top5}")
            worst5 = [(c, round(shelf_score[c], 2), round(shelf_dist_cache[c], 1))
                      for c in sorted_shelves[-5:]]
            print(f"[taps_slotting] Worst-5 shelves:                         {worst5}")





        cap_map = {cell: len(items) for cell, items in self.shelf_assignments.items()}

        new_assignment: Dict[Cell, list] = {cell: [] for cell in sorted_shelves}
        item_ptr = 0
        for cell in sorted_shelves:
            cap = cap_map[cell]
            for _ in range(cap):
                if item_ptr >= len(all_items):
                    break
                item = all_items[item_ptr]
                item["BFS_Distance"] = shelf_dist_cache[cell]
                new_assignment[cell].append(item)
                item_ptr += 1

        self.shelf_assignments = new_assignment

        if verbose:
            print(f"[taps_slotting] Items assigned: {item_ptr}/{len(all_items)}")


        if rebuild:
            if verbose:
                print("[taps_slotting] Rebuilding heatmap...")
            self.build_heatmap(orders)
            gini_after = self._gini_coefficient(self.cell_counts)
        else:
            gini_after = None

        result = {
            "gini_before": gini_before,
            "gini_after":  gini_after,
            "gini_delta":  (gini_after - gini_before) if gini_after is not None else None,
            "alpha":       alpha,
            "n_items":     item_ptr,
            "n_shelves":   len(sorted_shelves),
        }

        if verbose:
            print(f"[taps_slotting] α = {alpha}")
            print(f"[taps_slotting] Gini BEFORE: {gini_before:.4f}")
            if gini_after is not None:
                delta = gini_after - gini_before
                print(f"[taps_slotting] Gini AFTER:  {gini_after:.4f}  "
                      f"(Δ = {delta:+.4f}, {delta/gini_before*100:+.1f}%)")

        return result

    def hot_zone_relegation(
        self,
        orders: List[List[str]],
        hot_pct: float = 0.15,
        rebuild: bool = True,
        verbose: bool = True,
    ) -> dict:
        """Move C items into the hottest shelves to relieve congested aisles.

        Inputs: orders, hot share, rebuild flag, verbose flag.
        Returns: dict with Gini before and after, hot share parameters,
        and counts of moved items.
        """
        if self.shelf_assignments is None:
            raise ValueError("Call assign_items_to_shelves() first.")
        if self.heatmap is None:
            if verbose:
                print("[hot_zone_relegation] No heatmap available, building it...")
            self.build_heatmap(orders)

        gini_before = self._gini_coefficient(self.cell_counts)

        shelf_cong  = self._shelf_access_congestion()
        dist_corr   = self.corridor_distances()
        grid        = self.grid
        rows_, cols_ = grid.shape

        def bfs_d(cell: Cell) -> float:
            r, c = cell
            best = float('inf')
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= rr < rows_ and 0 <= cc < cols_ and grid[rr, cc] in (0, 2):
                    best = min(best, dist_corr[rr, cc])
            return best


        all_cells   = list(self.shelf_assignments.keys())
        n_hot       = max(1, int(len(all_cells) * hot_pct))
        sorted_by_cong = sorted(all_cells, key=lambda c: -shelf_cong[c])
        hot_cells   = set(sorted_by_cong[:n_hot])
        cool_cells  = [c for c in sorted_by_cong[n_hot:]]

        if verbose:
            print(f"[hot_zone_relegation] Hot cells: {len(hot_cells)}, "
                  f"Cool cells: {len(cool_cells)}")
            avg_hot_cong  = sum(shelf_cong[c] for c in hot_cells) / len(hot_cells)
            avg_cool_cong = sum(shelf_cong[c] for c in cool_cells) / max(len(cool_cells),1)
            print(f"[hot_zone_relegation] Avg congestion — hot: {avg_hot_cong:.4f}, "
                  f"cool: {avg_cool_cong:.4f}")


        all_items: list = []
        for items in self.shelf_assignments.values():
            all_items.extend(items)


        a_items = sorted(
            [i for i in all_items if i["ABC_XYZ"][0] == "A"],
            key=lambda x: -x.get("Total_Annual_Units", 0.0)
        )

        c_items = sorted(
            [i for i in all_items if i["ABC_XYZ"][0] == "C"],
            key=lambda x: x.get("Total_Annual_Units", 0.0)
        )

        b_items = sorted(
            [i for i in all_items if i["ABC_XYZ"][0] == "B"],
            key=lambda x: -x.get("Total_Annual_Units", 0.0)
        )




        new_assignment: Dict[Cell, list] = {cell: [] for cell in all_cells}
        cap_map = {cell: len(items) for cell, items in self.shelf_assignments.items()}


        c_ptr = 0
        for cell in sorted(hot_cells, key=lambda c: bfs_d(c)):
            for _ in range(cap_map[cell]):
                if c_ptr < len(c_items):
                    item = c_items[c_ptr]
                    item["BFS_Distance"] = bfs_d(cell)
                    new_assignment[cell].append(item)
                    c_ptr += 1


        cool_sorted = sorted(cool_cells,
                             key=lambda c: (shelf_cong[c], bfs_d(c)))


        remaining_pool = a_items + b_items + c_items[c_ptr:]
        pool_ptr = 0
        for cell in cool_sorted:
            for _ in range(cap_map[cell]):
                if pool_ptr < len(remaining_pool):
                    item = remaining_pool[pool_ptr]
                    item["BFS_Distance"] = bfs_d(cell)
                    new_assignment[cell].append(item)
                    pool_ptr += 1

        self.shelf_assignments = {k: v for k, v in new_assignment.items() if v}

        total_assigned = sum(len(v) for v in self.shelf_assignments.values())
        if verbose:
            print(f"[hot_zone_relegation] Total items assigned: {total_assigned}")
            print(f"[hot_zone_relegation] C-items in hot zone: {c_ptr}")
            print(f"[hot_zone_relegation] A-items in cool zone: {len(a_items)}")


        if rebuild:
            if verbose:
                print("[hot_zone_relegation] Rebuilding heatmap...")
            self.build_heatmap(orders)
            gini_after = self._gini_coefficient(self.cell_counts)
        else:
            gini_after = None

        result = {
            "gini_before":  gini_before,
            "gini_after":   gini_after,
            "gini_delta":   (gini_after - gini_before) if gini_after is not None else None,
            "hot_pct":      hot_pct,
            "n_hot_cells":  len(hot_cells),
            "c_in_hot":     c_ptr,
            "a_in_cool":    len(a_items),
        }

        if verbose:
            print(f"[hot_zone_relegation] Gini BEFORE: {gini_before:.4f}")
            if gini_after is not None:
                delta = gini_after - gini_before
                print(f"[hot_zone_relegation] Gini AFTER:  {gini_after:.4f}  "
                      f"(Δ = {delta:+.4f}, {delta/gini_before*100:+.1f}%)")

        return result

    def aisle_balanced_slotting(
        self,
        orders: List[List[str]],
        rebuild: bool = True,
        verbose: bool = True,
    ) -> dict:
        """Spread A items across several aisles using round-robin assignment.

        Inputs: orders, rebuild flag, verbose flag.
        Returns: dict with Gini before and after and the number of A, B, C
        aisles used. Aims to share traffic across corridors instead of
        concentrating it in one or two.
        """
        if self.shelf_assignments is None:
            raise ValueError("Call assign_items_to_shelves() first.")
        if self.heatmap is None:
            self.build_heatmap(orders)

        gini_before = self._gini_coefficient(self.cell_counts)

        dist_corr   = self.corridor_distances()
        grid        = self.grid
        rows_, cols_ = grid.shape

        def bfs_d(cell: Cell) -> float:
            r, c = cell
            best = float('inf')
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= rr < rows_ and 0 <= cc < cols_ and grid[rr, cc] in (0, 2):
                    best = min(best, dist_corr[rr, cc])
            return best



        from collections import defaultdict
        aisle_shelves: Dict[int, list] = defaultdict(list)
        for cell in self.shelf_assignments:
            aisle_shelves[cell[0]].append(cell)


        aisle_dist = {}
        for aisle_row, cells in aisle_shelves.items():
            dists = [bfs_d(c) for c in cells]
            finite = [d for d in dists if d < float('inf')]
            aisle_dist[aisle_row] = sum(finite) / len(finite) if finite else float('inf')


        sorted_aisles = sorted(aisle_shelves.keys(), key=lambda a: aisle_dist[a])

        if verbose:
            print(f"[aisle_balanced] Total aisles: {len(sorted_aisles)}")
            for a in sorted_aisles[:6]:
                print(f"  Row {a:>3}: {len(aisle_shelves[a]):>4} shelves, "
                      f"avg dist = {aisle_dist[a]:.1f}")


        total_shelves = sum(len(v) for v in aisle_shelves.values())



        n_a_aisles = max(3, int(len(sorted_aisles) * 0.25))
        a_aisles   = sorted_aisles[:n_a_aisles]
        b_aisles   = sorted_aisles[n_a_aisles: n_a_aisles + int(len(sorted_aisles) * 0.30)]
        c_aisles   = sorted_aisles[n_a_aisles + int(len(sorted_aisles) * 0.30):]

        if verbose:
            print(f"[aisle_balanced] A-aisles: {len(a_aisles)} rows "
                  f"(dist {aisle_dist[a_aisles[0]]:.0f}–{aisle_dist[a_aisles[-1]]:.0f})")
            print(f"[aisle_balanced] B-aisles: {len(b_aisles)} rows")
            print(f"[aisle_balanced] C-aisles: {len(c_aisles)} rows")


        def shelves_for_aisles(aisle_rows):
            slots = []

            shelves_by_aisle = [
                sorted(aisle_shelves[a], key=lambda c: bfs_d(c))
                for a in aisle_rows
            ]
            max_per_aisle = max(len(s) for s in shelves_by_aisle) if shelves_by_aisle else 0
            for col_idx in range(max_per_aisle):
                for aisle_list in shelves_by_aisle:
                    if col_idx < len(aisle_list):
                        slots.append(aisle_list[col_idx])
            return slots

        a_slots = shelves_for_aisles(a_aisles)
        b_slots = shelves_for_aisles(b_aisles)
        c_slots = shelves_for_aisles(c_aisles)


        all_items: list = []
        for items in self.shelf_assignments.values():
            all_items.extend(items)

        a_items = sorted([i for i in all_items if i["ABC_XYZ"][0] == "A"],
                         key=lambda x: -x.get("Total_Annual_Units", 0.0))
        b_items = sorted([i for i in all_items if i["ABC_XYZ"][0] == "B"],
                         key=lambda x: -x.get("Total_Annual_Units", 0.0))
        c_items = sorted([i for i in all_items if i["ABC_XYZ"][0] == "C"],
                         key=lambda x: -x.get("Total_Annual_Units", 0.0))


        cap_map = {cell: len(items) for cell, items in self.shelf_assignments.items()}
        new_assignment: Dict[Cell, list] = {cell: [] for cell in self.shelf_assignments}

        def fill_slots(slots, item_pool, overflow_pool):
            """Spread A items across several aisles using round-robin assignment.

            Inputs: orders, rebuild flag, verbose flag.
            Returns: dict with Gini before and after and the number of A, B, C aisles used.
            """
            pool = item_pool + overflow_pool
            ptr = 0
            for cell in slots:
                for _ in range(cap_map.get(cell, 1)):
                    if ptr < len(pool):
                        item = pool[ptr]
                        item["BFS_Distance"] = bfs_d(cell)
                        new_assignment[cell].append(item)
                        ptr += 1
            return pool[ptr:]

        leftover_a = fill_slots(a_slots, a_items, [])
        leftover_b = fill_slots(b_slots, b_items, leftover_a)
        fill_slots(c_slots, c_items, leftover_b)


        self.shelf_assignments = {k: v for k, v in new_assignment.items() if v}

        if verbose:
            assigned = sum(len(v) for v in self.shelf_assignments.values())
            print(f"[aisle_balanced] Items assigned: {assigned}/{len(all_items)}")


        if rebuild:
            if verbose:
                print("[aisle_balanced] Rebuilding heatmap...")
            self.build_heatmap(orders)
            gini_after = self._gini_coefficient(self.cell_counts)
        else:
            gini_after = None

        result = {
            "gini_before":  gini_before,
            "gini_after":   gini_after,
            "gini_delta":   (gini_after - gini_before) if gini_after is not None else None,
            "n_a_aisles":   len(a_aisles),
            "n_b_aisles":   len(b_aisles),
            "n_c_aisles":   len(c_aisles),
        }

        if verbose:
            print(f"[aisle_balanced] Gini BEFORE: {gini_before:.4f}")
            if gini_after is not None:
                delta = gini_after - gini_before
                print(f"[aisle_balanced] Gini AFTER:  {gini_after:.4f}  "
                      f"(Δ = {delta:+.4f}, {delta/gini_before*100:+.1f}%)")

        return result

    def affinity_aware_slotting(
        self,
        orders: List[List[str]],
        top_k_pairs: int = 150,
        max_swaps: int = 400,
        congestion_tol: float = 0.10,
        rebuild: bool = True,
        verbose: bool = True,
    ) -> dict:
        """Group co-ordered SKUs into the same aisle on top of Aisle-Balanced.

        Inputs: orders, number of top pairs to process, swap cap, congestion
        tolerance, rebuild flag, verbose flag.
        Returns: dict with Gini before and after, number of swaps performed,
        and parameters used.
        """
        if self.shelf_assignments is None:
            raise ValueError("Call assign_items_to_shelves() first.")


        if verbose:
            print("[aahms] Step 1: running Aisle-Balanced as initial placement...")
        ab_result = self.aisle_balanced_slotting(orders, rebuild=True, verbose=False)
        gini_before = ab_result["gini_before"]

        if self.heatmap is None:
            self.build_heatmap(orders)


        if verbose:
            print("[aahms] Step 2: building co-occurrence matrix...")
        from collections import defaultdict, Counter
        cooccur: Dict[str, Dict[str, int]] = defaultdict(Counter)
        for order in orders:
            skus = list(dict.fromkeys(order))
            for idx_a in range(len(skus)):
                for idx_b in range(idx_a + 1, len(skus)):
                    a, b = skus[idx_a], skus[idx_b]
                    cooccur[a][b] += 1
                    cooccur[b][a] += 1


        pairs_scored = []
        seen = set()
        for sku_a, partners in cooccur.items():
            for sku_b, cnt in partners.items():
                key = tuple(sorted([sku_a, sku_b]))
                if key not in seen:
                    seen.add(key)
                    pairs_scored.append((cnt, sku_a, sku_b))
        pairs_scored.sort(reverse=True)
        top_pairs = pairs_scored[:top_k_pairs]

        if verbose:
            print(f"[aahms]   Unique pairs: {len(pairs_scored)}, "
                  f"using top {len(top_pairs)}")
            if top_pairs:
                print(f"[aahms]   Highest co-occurrence: {top_pairs[0][0]} "
                      f"({top_pairs[0][1]} ↔ {top_pairs[0][2]})")


        grid = self.grid
        rows_, cols_ = grid.shape
        dist_corr = self.corridor_distances()
        shelf_cong = self._shelf_access_congestion()

        def bfs_d(cell: Cell) -> float:
            r, c = cell
            best = float('inf')
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= rr < rows_ and 0 <= cc < cols_ and grid[rr, cc] in (0, 2):
                    best = min(best, dist_corr[rr, cc])
            return best


        sku_to_cell: Dict[str, Cell] = {}
        for cell, items in self.shelf_assignments.items():
            for item in items:
                sku_to_cell[item["Item_ID"]] = cell


        def aisle_of(cell: Cell) -> int:
            return cell[0]


        n_swaps = 0
        n_considered = 0

        for cnt, sku_a, sku_b in top_pairs:
            if n_swaps >= max_swaps:
                break
            cell_a = sku_to_cell.get(sku_a)
            cell_b = sku_to_cell.get(sku_b)
            if cell_a is None or cell_b is None:
                continue
            if aisle_of(cell_a) == aisle_of(cell_b):
                continue


            item_a = next((it for it in self.shelf_assignments[cell_a]
                           if it["Item_ID"] == sku_a), None)
            item_b = next((it for it in self.shelf_assignments[cell_b]
                           if it["Item_ID"] == sku_b), None)
            if item_a is None or item_b is None:
                continue

            abc_a = item_a["ABC_XYZ"][0]
            abc_b = item_b["ABC_XYZ"][0]
            target_aisle = aisle_of(cell_a)


            best_partner: Optional[Cell] = None
            best_score = float('inf')

            for cell_k, items_k in self.shelf_assignments.items():
                if not items_k:
                    continue
                item_k = items_k[0]
                if item_k["ABC_XYZ"][0] != abc_b:
                    continue
                if item_k["Item_ID"] == sku_a or item_k["Item_ID"] == sku_b:
                    continue
                if aisle_of(cell_k) != target_aisle:
                    continue

                n_considered += 1


                dist_k = bfs_d(cell_k)
                dist_b = bfs_d(cell_b)
                if dist_k > dist_b + 2:
                    continue


                cong_k = shelf_cong.get(cell_k, 0.0)
                cong_b = shelf_cong.get(cell_b, 0.0)
                if cong_k > cong_b * (1 + congestion_tol) + 1e-9:
                    continue


                score = cong_k + 0.01 * dist_k
                if score < best_score:
                    best_score = score
                    best_partner = cell_k

            if best_partner is None:
                continue


            cell_k = best_partner
            item_k = self.shelf_assignments[cell_k][0]


            item_b["BFS_Distance"] = bfs_d(cell_k)
            item_k["BFS_Distance"] = bfs_d(cell_b)

            self.shelf_assignments[cell_k] = [item_b]
            self.shelf_assignments[cell_b] = [item_k]

            sku_to_cell[sku_b]              = cell_k
            sku_to_cell[item_k["Item_ID"]]  = cell_b

            n_swaps += 1

        if verbose:
            print(f"[aahms] Step 4: {n_swaps} swaps performed "
                  f"(considered {n_considered} candidates)")


        if rebuild:
            if verbose:
                print("[aahms] Step 5: rebuilding heatmap...")
            self.build_heatmap(orders)
            gini_after = self._gini_coefficient(self.cell_counts)
        else:
            gini_after = None

        result = {
            "gini_before":   gini_before,
            "gini_after":    gini_after,
            "gini_delta":    (gini_after - gini_before) if gini_after is not None else None,
            "n_swaps":       n_swaps,
            "top_k_pairs":   top_k_pairs,
            "n_ab_a_aisles": ab_result["n_a_aisles"],
        }

        if verbose:
            print(f"[aahms] Gini BEFORE (after AB): {gini_before:.4f}")
            if gini_after is not None:
                delta = gini_after - gini_before
                print(f"[aahms] Gini AFTER (AAHMS):    {gini_after:.4f}  "
                      f"(Δ = {delta:+.4f}, {delta/gini_before*100:+.1f}%)")

        return result

