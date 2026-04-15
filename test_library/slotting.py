from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd

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
        """Place items on shelves in order of ABC_XYZ rank and annual units.

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
