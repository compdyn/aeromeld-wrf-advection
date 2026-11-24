#!/usr/bin/env python
"""
Convert affx.py checkpoints (.pt) to numpy-only .npz files, so the models can
be used without torch: every tensor is stored as a float32/int array under its
own key, and every other entry (flags such as dec, mono, bounded; hyper-
parameters) as JSON in the '__meta__' entry.

Usage: python pt_to_npz.py IN.pt [IN.pt ...]   (writes IN.npz beside each)
"""
import json
import sys

import numpy as np
import torch

for f in sys.argv[1:]:
    st = torch.load(f, map_location='cpu', weights_only=False)
    arrays, meta = {}, {}
    for k, v in st.items():
        if torch.is_tensor(v):
            arrays[k] = v.detach().cpu().numpy()
        elif isinstance(v, np.ndarray):
            arrays[k] = v
        else:
            try:
                json.dumps(v); meta[k] = v
            except TypeError:
                meta[k] = repr(v)
    out = f[:-3] + '.npz'
    np.savez_compressed(out, __meta__=np.array(json.dumps(meta)), **arrays)
    print(out, len(arrays), 'arrays', sorted(meta)[:12])
