# Vendored modules

Copies of code that lives outside any repository the paper can point to, so
that the encoder and decoder run anywhere `partnn`, PyPartMC and torch are
installed. `extract/aeromeld.py` (`vae_utils()`) imports `vae_partmc_utils`
from here, so the figure products and the WRF-AeroMELD inputs
(`inputs/encode_populations.py`) use this one copy.

| File | Copied 2026-11-10 from |
|---|---|
| `vae_partmc_utils.py` | the authors' working copy of `code_partnn/notebooks/vae_partmc_utils.py` (not in a repository) |
| `extract_diagnostics.py` | the authors' `analyze_particles/extract_diagnostics.py` |
| `extract_diagnostics_part.py` | the authors' `analyze_particles/extract_diagnostics_part.py` |

The copies differ from the originals only in the default data paths, which
are placeholders here (none is used by this repository), and in one line of
`vae_partmc_utils.py`: it imports the two `extract_diagnostics*` modules from
its own directory (`sys.path.insert(0, str(_HERE))`), not from an absolute
`analyze_particles` path.

`vae_partmc_utils` still needs the `partnn` package (`partnn.io_utils`,
`partnn.notebooks.n22_utils`; both tracked in `code_partnn`). Its defaults
`FPIDX` / `RESDIR` are not used by this repository: every caller passes
`fpidx=` and `resdir=config.MODEL_RESDIR`.

`partnn_io_cfg.patch` is an uncommitted change to `code_partnn`'s
`partnn/io_cfg.py` (diffed against commit `04a2235`). The committed version
asserts that the current working directory contains `code_partnn`, so any
import of partnn from this repository fails. With the patch, the project root
is taken from the package's own location. Apply it to your partnn checkout
(`git apply partnn_io_cfg.patch` inside `code_partnn`) until it is committed
upstream.

To re-sync after the originals change, copy them again, reapply the one-line
import change, and update the table.
