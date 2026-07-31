import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import click
from model.file_features import score_file

@click.command()
@click.argument('root', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True)
def main(root):
    root = os.path.abspath(root)

    print(f"Analyzing ROOT directory: {root}")

    SKIP = {'.git', '__pycache__', 'node_modules', 'build', 'dist'}
    files = []
    for dp, dn, fns in os.walk(root):
        dn[:] = [d for d in dn if d not in SKIP]
        for f in fns:
            files.append(os.path.relpath(os.path.join(dp, f), root))

    scores = [(score_file(f), f) for f in files]
    scores.sort(reverse=True)

    print("=" * 65)
    print("SCORE DISTRIBUTION")
    print(f"{'bucket':>12}  {'count':>6}  {'%':>5}  {'cum%':>5}")
    print("-" * 65)
    buckets = {}
    for s, _ in scores:
        b = round(int(s * 10) / 10, 1)
        buckets[b] = buckets.get(b, 0) + 1
    cum_pct = 0.0
    for b in sorted(buckets, reverse=True):
        bar = '█' * int(buckets[b] / len(scores) * 80)
        pct = buckets[b] / len(scores) * 100
        cum_pct += pct
        marker = ' ← threshold' if b == 0.4 else ''
        print(f"  {b:>4.1f}-{b+0.1:.1f}   {buckets[b]:6d}  {pct:5.1f}%  {cum_pct:5.1f}%  {bar}{marker}")

    # ── Section 2: Files right around the 0.45 boundary ───────────────────────
    THRESHOLDS = [0.55, 0.50, 0.45, 0.40, 0.35]

    for thr in THRESHOLDS:
        lo, hi = thr - 0.05, thr + 0.05
        band = [(s, f) for s, f in scores if lo <= s < hi]
        print(f"\n{'='*65}")
        print(f"BAND [{lo:.2f}, {hi:.2f})  — {len(band)} files  (threshold at {thr})")
        print(f"{'score':>6}  path")
        print("-" * 65)
        for s, f in band[:25]:
            print(f"  {s:.2f}  {f}")
        if len(band) > 25:
            print(f"  ... and {len(band)-25} more")

    # ── Section 3: What gets included/excluded at different thresholds ─────────
    print(f"\n{'='*65}")
    print("THRESHOLD SENSITIVITY")
    print(f"{'thr':>5}  {'include':>8}  {'exclude':>8}  sample of borderline INCLUDED files")
    print("-" * 65)
    for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        inc = [(s, f) for s, f in scores if s >= thr]
        exc = [(s, f) for s, f in scores if s < thr]
        # show the 3 lowest-scoring included files (most likely false positives)
        examples = ', '.join(os.path.basename(f) for _, f in inc[-3:]) if inc else '-'
        print(f"  {thr:.2f}  {len(inc):8d}  {len(exc):8d}  {examples}")


if __name__ == '__main__':
    main()

