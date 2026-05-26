"""Backward compatibility shim.

The implementation now lives in the warehouse package. This file is kept so
that existing imports of the form `from warehouse_model import WarehouseModel`
in notebooks and scripts keep working without changes.
"""
from warehouse import WarehouseModel, Cell, WAREHOUSE_SEED, ORDER_SEED

__all__ = ["WarehouseModel", "Cell", "WAREHOUSE_SEED", "ORDER_SEED"]
