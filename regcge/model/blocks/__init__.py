"""Equation blocks.  Each module exposes ``declare(M)`` (variables and
parameters) and ``equations(M)``.  ``M`` is the :class:`RegionalCGE` under
construction: ``M.m`` the CoreModel, ``M.c`` the Calibration, ``M.S`` the sets
and ``M.x`` a dict of shared expressions (purchaser prices, tax wedges, ...).

To modify the model, edit a block or pass a custom list of block modules to
``RegionalCGE(..., blocks=[...])``.
"""
