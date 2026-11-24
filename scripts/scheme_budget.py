#!/usr/bin/env python
"""The error budget of each latent-transport scheme, for every seed of the main and affine models.

From the offline replays all0..all9 (transport/, every scheme for ten seeds of
twin_k16_of and ax_rel1_of): paperfigs.transport_schemes.budget at 04 UTC
against the exact answer. Terms: the transport scheme, decoding mixtures,
end-member; they add up to the total. Writes data/scheme_budget.json
(committed), which plot/transport_schemes.py (schemes_budget) reads.

Usage (from scripts/): python scheme_budget.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paperfigs import config  # noqa: E402
from paperfigs import transport_schemes as T  # noqa: E402

OUT = config.DATA_DIR / 'scheme_budget.json'
MODELS = ('twin_k16_of', 'ax_rel1_of')
SEEDS = range(10)


def main():
    res = {}
    for m in MODELS:
        for k in SEEDS:
            model = '%s.%d.0' % (m, k)
            rm = T.run_model(model)
            keys = ['donor', 'shB:' + rm, 'sep:' + rm, 'oracle']
            b = T.budget(model, 'all%d' % k, keys=keys)
            res[model] = {key.split(':')[0]: v for key, v in b.items()}
            print(model, '  '.join('%s %s' % (key.split(':')[0], '/'.join('%.0f' % (100 * b[key][g]['error'])
                                                                        for g in b[key])) for key in keys), flush=True)
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
