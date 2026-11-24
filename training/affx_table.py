#!/usr/bin/env python
"""
One line per affx.py run (mean over its seeds): the paper's geometric mean
and eight test errors, the full-range frozen-fraction error, the share of
test populations with any negative value (all and per diagnostic) and with
any mix_eval validity flag, and the 16-way mixture geometric mean.

Usage: python affx_table.py DIR [DIR ...] [--md]
"""
import glob
import json
import sys

md = '--md' in sys.argv
dirs = [a for a in sys.argv[1:] if a != '--md']
cols = [('geo', 'geo'), ('ccn_cdf', 'CCN'), ('logqs_pop', 'scat'), ('logqa_pop', 'abs'),
        ('logfrznfrac_tmp', 'FF'), ('ff_full', 'FFall'), ('m_chmprthst', 'Mspec'), ('n_prthst', 'N'),
        ('m_prthst', 'Mtot'), ('m_chmprt', 'Mbulk'), ('neg_any', 'neg'), ('neg_M', 'negM'),
        ('neg_N', 'negN'), ('neg_S', 'negS'), ('neg_A', 'negA'), ('neg_F', 'negF'), ('viol_any', 'viol'),
        ('ccn_gt1', 'C>1'), ('ff_nonmono', 'FFmono')]
rows = []
for d in dirs:
    for f in sorted(glob.glob(f'{d}/*.json')):
        r = json.load(open(f))
        t, m = r['mean']['test'], r['mean']['mix16']
        rows.append([r['args']['name'], len(r['seeds'])] + [100 * t[k] for k, _ in cols] + [100 * m['geo'], r['wall_s'] / 60])
hdr = ['run', 'n'] + [h for _, h in cols] + ['mix16', 'min']
if md:
    print('| ' + ' | '.join(hdr) + ' |'); print('|' + '---|' * len(hdr))
    for r in rows:
        print('| ' + ' | '.join([r[0], str(r[1])] + [f'{v:.1f}' for v in r[2:]]) + ' |')
else:
    w = max(len(r[0]) for r in rows) if rows else 10
    print(f'{"run":{w}s} {"n":>2s} ' + ' '.join(f'{h:>6s}' for h in hdr[2:]))
    for r in rows:
        print(f'{r[0]:{w}s} {r[1]:2d} ' + ' '.join(f'{v:6.1f}' for v in r[2:]))
