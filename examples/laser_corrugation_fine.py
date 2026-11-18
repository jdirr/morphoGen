"""The corrugations about the angle of the rule (docs/laser.md, §17.5): ``θ*`` = 59.5°
places the flanks at ``H/P`` = tan θ*/(2π) = 0.270. Periodic sheets of corrugated laser
martensite, ``P`` = 20 mm, 24 % treated, ``H/P`` = 0.26, 0.27 and 0.28 under the ductile
and graded rules, on elements of 0.5 mm (between the heights of
``examples/laser_geometric_hardening.py``). Writes ``results/laser_corrugation_fine``
(``.csv`` and ``.npz``, the format of that study)."""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import laser_geometric_hardening as g

OUT = Path("results/laser_corrugation_fine")
RATIOS = (0.26, 0.27, 0.28)


def main() -> None:
    jobs = [
        g._job("sheet", rule, g.MARTENSITE, 0.24, 20.0, (r,), 0.5)
        for rule in ("ductile", "graded")
        for r in RATIOS
    ]
    with ProcessPoolExecutor(len(jobs), initializer=g._init) as pool:
        rows = list(pool.map(g.evaluate, jobs))
    g.save({r["key"]: r for r in rows}, OUT)
    for r in rows:
        print(
            f"{r['rule']:8s} H/P {r['ratios']}: Rm {r['tensile_strength']:.0f}, "
            f"Ag {100 * r['uniform_elongation']:.1f}, fail {100 * r['failure_strain']:.1f}, "
            f"E {r['energy']:.0f}",
            flush=True,
        )
    print("written:", OUT.with_suffix(".csv"))


if __name__ == "__main__":
    main()
