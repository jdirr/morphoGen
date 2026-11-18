"""Gather the dog-bones of crossed tracks (``results/laser_crossed_specimen*.log``,
``results/laser_crossed_edges.log``) into ``results/laser_crossed_specimens.csv``."""

from __future__ import annotations

import csv
import re
from pathlib import Path

LINE = re.compile(
    r"±(?P<spread>[\d.]+)° (?P<state>[\w-]+) (?P<length>[\d.]+) mm, treated (?P<treated>[\d.]+): "
    r"Rm (?P<rm>\d+), Ag (?P<ag>[\d.]+), fail (?P<fail>[\d.]+), E (?P<energy>\d+)"
)
WIDE = re.compile(
    r"±(?P<spread>[\d.]+)° (?P<state>[\w-]+) (?P<length>[\d.]+) mm, width (?P<width>[\d.]+) mm: "
    r"Rm (?P<rm>\d+), Ag (?P<ag>[\d.]+), E (?P<energy>\d+)"
)
EDGES = re.compile(
    r"±(?P<spread>[\d.]+)° network (?P<state>[\w-]+), edges (?P<edge>[\w-]+): "
    r"Rm (?P<rm>\d+), Ag (?P<ag>[\d.]+), fail (?P<fail>[\d.]+), E (?P<energy>\d+)"
)


def main() -> None:
    rows = {}
    for path in sorted(Path("results").glob("laser_crossed_specimen*.log")):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = LINE.search(line)
            if m:
                d = m.groupdict()
                d["edge"], d["width"] = "free", "44"
                rows[(d["spread"], d["state"], d["length"], "free")] = d
    wide = Path("results/laser_crossed_wide.log")
    if wide.exists():
        for line in wide.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = WIDE.search(line)
            if m:
                d = m.groupdict()
                d["edge"], d["treated"], d["fail"] = "free", "", d["ag"]
                rows[(d["spread"], d["state"], d["length"], f"free, {d['width']} mm")] = d
    edges = Path("results/laser_crossed_edges.log")
    if edges.exists():
        for line in edges.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = EDGES.search(line)
            if m:
                d = m.groupdict()
                d["length"], d["treated"], d["width"] = "90", "", "44"
                rows[(d["spread"], d["state"], "90", d["edge"])] = d
    out = Path("results/laser_crossed_specimens.csv")
    fields = ("spread", "state", "length", "width", "edge", "treated", "rm", "ag", "fail", "energy")
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fields, extrasaction="ignore")
        w.writeheader()
        for key in sorted(rows, key=lambda k: (float(k[0]), k[1], float(k[2]), k[3])):
            w.writerow(rows[key])
    print(f"written: {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
