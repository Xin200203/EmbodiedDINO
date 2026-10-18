#!/usr/bin/env python3
"""Stack a scene's GT / matched-host / EmbodiedDINO panels into one strip.

Judging three separate files by flipping between them does not work: the eye
cannot hold an instance decomposition in memory long enough to compare it.  Side
by side at equal height, a fragmented object is obvious.

Panels are padded, never scaled to a common box: the renderer already crops all
three of a triplet to the same window, so equal-height padding preserves the
one-to-one pixel correspondence that makes the comparison trustworthy.
"""

import argparse
import glob
import os
import os.path as osp

from PIL import Image, ImageDraw

ORDER = ('gt', 'host', 'ours')
LABEL = {'gt': 'Ground truth', 'host': 'Matched host', 'ours': 'EmbodiedDINO'}


def strip(paths, gap, label_h, title):
    ims = [Image.open(p).convert('RGB') for p in paths]
    h = max(i.height for i in ims)
    w = sum(i.width for i in ims) + gap * (len(ims) - 1)
    out = Image.new('RGB', (w, h + label_h), 'white')
    d = ImageDraw.Draw(out)
    x = 0
    for p, im in zip(paths, ims):
        out.paste(im, (x, label_h + (h - im.height) // 2))
        key = osp.basename(p).rsplit('_', 1)[-1][:-4]
        d.text((x + 6, 4), LABEL.get(key, key), fill=(20, 20, 20))
        x += im.width + gap
    return out                      # the strip's filename carries the scene/view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--gap', type=int, default=8)
    ap.add_argument('--label-h', type=int, default=18)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    groups = {}
    for p in glob.glob(osp.join(a.root, '**', '*.png'), recursive=True):
        b = osp.basename(p)[:-4]
        panel = b.rsplit('_', 1)[-1]
        if panel not in ORDER:
            continue
        # The key must carry the sub-directory: sweeps write the SAME filename
        # under e55/ e72/ e90/, and keying on the basename alone silently merged
        # three elevations into one triplet.
        rel = osp.relpath(osp.dirname(p), a.root).replace(osp.sep, '_')
        key = (rel + '_' if rel not in ('.', '') else '') + b.rsplit('_', 1)[0]
        groups.setdefault(key, {})[panel] = p

    n = 0
    for key, d in sorted(groups.items()):
        if not all(k in d for k in ORDER):
            print(f'[skip] {key}: only {sorted(d)}')
            continue
        im = strip([d[k] for k in ORDER], a.gap, a.label_h, key)
        im.save(osp.join(a.out, key + '.png'))
        n += 1
    print(f'[write] {n} strips -> {a.out}')


if __name__ == '__main__':
    main()
