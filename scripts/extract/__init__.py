"""Builders for the processed data products (see paperfigs/products.py).

Each module's build functions read raw model output and return
``(xarray.Dataset, [source paths])``. They are imported only when a product is
built, so heavy dependencies (torch, partnn, PyPartMC) are imported inside the
functions that need them, never at module level.
"""
