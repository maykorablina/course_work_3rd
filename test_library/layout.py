from __future__ import annotations

from typing import Optional
import numpy as np

from .constants import Cell


class LayoutMixin:
    def generate(self) -> np.ndarray:
        """Build the warehouse grid with regular shelf rows and one door.

        Inputs: object attributes rows, cols, shelf parameters, margin, door.
        Returns: numpy array of shape (rows, cols) with codes 0 (aisle), 1 (shelf), 2 (door).
        """
        grid = np.zeros((self.rows, self.cols), dtype=int)

        row_step = self.shelf_rows_height + self.aisle_width

        r = self.margin
        while r + self.shelf_rows_height <= self.rows - self.margin:

            c = self.margin
            while c + self.shelf_length <= self.cols - self.margin:

                for rr in range(r, r + self.shelf_rows_height):
                    for cc in range(c, c + self.shelf_length):
                        grid[rr, cc] = 1
                c += self.shelf_length + self.shelf_gap
            r += row_step

        dr, dc = self.door
        grid[dr, dc] = 0
        grid[dr, dc] = 2

        self.grid = grid
        return grid
