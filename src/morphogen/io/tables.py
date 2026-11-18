"""Tables of results as raw data files: comma-separated values (``.csv``), text in
tab-separated columns under a ``#`` header (``.txt``, ``.tsv``: ``numpy.loadtxt`` reads
them) and Excel workbooks (``.xlsx``), without dependency: a workbook is a zip of a few
XML parts (Office Open XML), its strings inline, its header row bold and frozen.

A table is a list of records (dicts), a mapping of columns (name -> values) or a 2D array
(with ``header``); several tables (a mapping name -> table) become the sheets of a
workbook, or one text file each.
"""

from __future__ import annotations

import csv
import math
import zipfile
from pathlib import Path
from xml.etree import ElementTree
from xml.sax.saxutils import escape

import numpy as np

FORMATS = {".csv": "csv", ".txt": "txt", ".tsv": "txt", ".dat": "txt", ".xlsx": "xlsx"}


def as_columns(table, header=None) -> tuple[list[str], list[list]]:
    """The header and the columns (lists) of a table (records, columns or an array)."""
    if isinstance(table, dict):
        names = [str(k) for k in table]
        columns = [
            list(np.ravel(np.asarray(v, dtype=object))) if np.ndim(v) else [v]
            for v in table.values()
        ]
    elif isinstance(table, (list, tuple)) and table and isinstance(table[0], dict):
        names = []
        for row in table:
            names.extend(k for k in row if k not in names)
        columns = [[row.get(k, "") for row in table] for k in names]
    else:
        data = np.atleast_2d(np.asarray(table, dtype=object))
        names = list(header) if header is not None else [f"c{j + 1}" for j in range(data.shape[1])]
        columns = [list(data[:, j]) for j in range(data.shape[1])]
    if header is not None and len(header) == len(names):
        names = [str(h) for h in header]
    n = max((len(c) for c in columns), default=0)
    columns = [c + [""] * (n - len(c)) for c in columns]
    return names, columns


def _plain(value):
    """A cell value as a Python number, bool or string."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return "" if value is None else str(value)


def _text(value, precision: int | None) -> str:
    value = _plain(value)
    if isinstance(value, float):
        return repr(value) if precision is None else f"{value:.{precision}g}"
    return str(value)


def _column_name(j: int) -> str:
    out = ""
    j += 1
    while j:
        j, r = divmod(j - 1, 26)
        out = chr(65 + r) + out
    return out


def _sheet_xml(names, columns) -> str:
    rows = ['<row r="1">'] + [
        f'<c r="{_column_name(j)}1" t="inlineStr" s="1"><is><t>{escape(n)}</t></is></c>'
        for j, n in enumerate(names)
    ]
    rows.append("</row>")
    for i in range(len(columns[0]) if columns else 0):
        r = i + 2
        cells = [f'<row r="{r}">']
        for j, column in enumerate(columns):
            value, ref = _plain(column[i]), f"{_column_name(j)}{r}"
            if isinstance(value, bool):
                cells.append(f'<c r="{ref}" t="b"><v>{int(value)}</v></c>')
            elif isinstance(value, (int, float)):
                if isinstance(value, float) and not math.isfinite(value):
                    cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{value}</t></is></c>')
                else:
                    cells.append(f'<c r="{ref}"><v>{value!r}</v></c>')
            elif value != "":
                cells.append(
                    f'<c r="{ref}" t="inlineStr"><is>'
                    f'<t xml:space="preserve">{escape(value)}</t></is></c>'
                )
        cells.append("</row>")
        rows.extend(cells)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
        'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        "<sheetData>" + "".join(rows) + "</sheetData></worksheet>"
    )


def _write_xlsx(path: Path, sheets: dict) -> None:
    names = list(sheets)
    main = "application/vnd.openxmlformats-officedocument.spreadsheetml"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{k + 1}.xml" ContentType="{main}.worksheet+xml"/>'
        for k in range(len(names))
    )
    content = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f'<Override PartName="/xl/workbook.xml" ContentType="{main}.sheet.main+xml"/>'
        f'<Override PartName="/xl/styles.xml" ContentType="{main}.styles+xml"/>'
        + overrides
        + "</Types>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'<Relationship Id="rId1" Type="{rel}/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    sheets_xml = "".join(
        f'<sheet name="{escape(str(n)[:31])}" sheetId="{k + 1}" r:id="rId{k + 1}"/>'
        for k, n in enumerate(names)
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        f'xmlns:r="{rel}"><sheets>{sheets_xml}</sheets></workbook>'
    )
    links = "".join(
        f'<Relationship Id="rId{k + 1}" Type="{rel}/worksheet" '
        f'Target="worksheets/sheet{k + 1}.xml"/>'
        for k in range(len(names))
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + links
        + f'<Relationship Id="rId{len(names) + 1}" Type="{rel}/styles" '
        'Target="styles.xml"/></Relationships>'
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
        '<font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="2"><fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1">'
        '<xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
        '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs>'
        "</styleSheet>"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("xl/workbook.xml", workbook)
        z.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        z.writestr("xl/styles.xml", styles)
        for k, n in enumerate(names):
            z.writestr(f"xl/worksheets/sheet{k + 1}.xml", _sheet_xml(*sheets[n]))


def write_table(
    path, table, *, header=None, sheet: str = "data", precision: int | None = None
) -> Path:
    """Write a table to ``.csv``, ``.txt``/``.tsv``/``.dat`` or ``.xlsx`` (by its suffix);
    ``precision`` significant digits for the text formats (all by default)."""
    path = Path(path)
    kind = FORMATS.get(path.suffix.lower())
    if kind is None:
        raise ValueError(f"write_table: a suffix among {sorted(FORMATS)}")
    names, columns = as_columns(table, header)
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "xlsx":
        _write_xlsx(path, {sheet: (names, columns)})
    elif kind == "csv":
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(names)
            for row in zip(*columns, strict=True):
                w.writerow([_text(v, precision) for v in row])
    else:
        with path.open("w", encoding="utf-8") as f:
            f.write("# " + "\t".join(names) + "\n")
            for row in zip(*columns, strict=True):
                f.write("\t".join(_text(v, precision) for v in row) + "\n")
    return path


def write_tables(path, tables: dict, *, precision: int | None = None) -> list[Path]:
    """Several tables: the sheets of one ``.xlsx`` workbook, or one text file each
    (``<stem>_<name><suffix>``)."""
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_xlsx(path, {name: as_columns(t) for name, t in tables.items()})
        return [path]
    return [
        write_table(path.with_name(f"{path.stem}_{name}{path.suffix}"), t, precision=precision)
        for name, t in tables.items()
    ]


def _number(text: str):
    try:
        value = float(text)
    except ValueError:
        return text
    return (
        int(value) if value.is_integer() and "." not in text and "e" not in text.lower() else value
    )


def read_table(path, sheet: int = 0) -> dict[str, list]:
    """The columns (name -> values) of a table written by :func:`write_table` (or any
    simple ``.csv``, text or ``.xlsx`` table with a header row)."""
    path = Path(path)
    kind = FORMATS.get(path.suffix.lower())
    if kind == "csv":
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        names, body = rows[0], rows[1:]
        return {n: [_number(r[j]) for r in body] for j, n in enumerate(names)}
    if kind == "txt":
        lines = path.read_text(encoding="utf-8").splitlines()
        names = lines[0].lstrip("#").strip().split("\t")
        body = [line.split("\t") for line in lines[1:] if line.strip()]
        return {n: [_number(r[j]) for r in body] for j, n in enumerate(names)}
    if kind != "xlsx":
        raise ValueError(f"read_table: a suffix among {sorted(FORMATS)}")
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ElementTree.fromstring(z.read("xl/sharedStrings.xml"))
            shared = [
                "".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t"))
                for si in root.findall("m:si", ns)
            ]
        root = ElementTree.fromstring(z.read(f"xl/worksheets/sheet{sheet + 1}.xml"))
    grid = []
    for row in root.iter(f"{{{ns['m']}}}row"):
        values = {}
        for c in row.findall("m:c", ns):
            ref, kind_c = c.get("r"), c.get("t")
            col = 0
            for ch in "".join(x for x in ref if x.isalpha()):
                col = 26 * col + ord(ch) - 64
            if kind_c == "inlineStr":
                value = "".join(t.text or "" for t in c.iter(f"{{{ns['m']}}}t"))
            elif kind_c == "s":
                value = shared[int(c.find("m:v", ns).text)]
            elif kind_c == "b":
                value = c.find("m:v", ns).text == "1"
            else:
                v = c.find("m:v", ns)
                value = _number(v.text) if v is not None else ""
            values[col - 1] = value
        grid.append(values)
    names = [str(grid[0][j]) for j in sorted(grid[0])]
    return {n: [r.get(j, "") for r in grid[1:]] for j, n in enumerate(names)}
