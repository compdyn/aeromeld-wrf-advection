"""Shared configuration, registry and product I/O for the paper's figures.

See scripts/PLAN.md for the design. In short: extract/ turns raw model output
into small NetCDF products under data/processed/, and plot/ turns products into
the PDFs in figures/. ``products.load`` joins the two, building a
product from the raw output when it is missing or stale.
"""
