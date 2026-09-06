"""Extract benchmark tables from the public PINNacle arXiv source.

The parser intentionally targets the paper's semantic table structure rather
than page coordinates. This makes provenance auditable and exposes duplicate or
conflicting claims without copying the paper into this repository.
"""

from __future__ import annotations

import io
import re
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ARXIV_SOURCE_URL = "https://export.arxiv.org/e-print/2306.08827v2"
ARXIV_ID = "2306.08827v2"

STYLE_RE = re.compile(
    r"\\(?:cellcolor\{[^}]+\}|textbf\{|multicolumn\{[^}]+\}\{[^}]+\}\{)"
)
NUMBER_RE = re.compile(
    r"(?P<mean>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:E[+-]?\d+)?)"
    r"(?:\((?P<std>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:E[+-]?\d+)?)\))?",
    re.IGNORECASE,
)


def download_tex(url: str = ARXIV_SOURCE_URL) -> str:
    """Download the versioned arXiv source and return its main TeX document."""
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        member = archive.getmember("main_iclr.tex")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise FileNotFoundError("main_iclr.tex was not found in the arXiv source")
        return extracted.read().decode("utf-8")


def _clean(cell: str) -> str:
    cell = STYLE_RE.sub("", cell)
    cell = cell.replace("\\textbf", "").replace("\\hline", "")
    cell = cell.replace("\\", "").replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", cell).strip()


def _table_before(tex: str, marker: str, occurrence: int = 1) -> str:
    positions = [m.start() for m in re.finditer(re.escape(marker), tex)]
    if len(positions) < occurrence:
        raise ValueError(f"Could not find occurrence {occurrence} of {marker}")
    marker_pos = positions[occurrence - 1]
    start = tex.rfind(r"\begin{tabular}", 0, marker_pos)
    end = tex.find(r"\end{tabular}", start)
    if start < 0 or end < 0:
        raise ValueError(f"Could not isolate table before {marker}")
    return tex[start : end + len(r"\end{tabular}")]


def _parse_table(table: str, source: str) -> pd.DataFrame:
    # Some published rows wrap across physical TeX lines. Parse semantic row
    # terminators so wrapped component values cannot become phantom PDE cases.
    lines = [
        re.sub(r"\s+", " ", row).strip()
        for row in re.split(r"\\\\(?:hline)?", table)
        if "&" in row
    ]
    metric_line = next(line for line in lines if re.search(r"(?:L[12]RE|mERR|MSE)\s*&", line))
    metric = re.search(r"(L[12]RE|mERR|MSE)\s*&", metric_line).group(1)
    header_index = next(i for i, line in enumerate(lines) if "-- &" in line)
    method_cells = [_clean(c) for c in lines[header_index].split("&")][2:]
    methods = [m for m in method_cells if m and m not in {"-", "--"}]

    records: list[dict[str, object]] = []
    current_family = ""
    for line in lines[header_index + 1 :]:
        if "caption" in line:
            break
        cells = [_clean(c) for c in line.split("&")]
        if len(cells) < 3:
            continue
        family_cell, case = cells[0], cells[1]
        if family_cell:
            multi = re.search(r"multirow\d+\*([A-Za-z ]+)", family_cell)
            current_family = (multi.group(1) if multi else family_cell).strip()
        if not case:
            continue
        values = cells[2 : 2 + len(methods)]
        for method, raw in zip(methods, values):
            normalized = raw.strip()
            if normalized.lower().startswith("nan") or normalized in {"-", "--", "–"}:
                mean = std = np.nan
            else:
                match = NUMBER_RE.search(normalized)
                if not match:
                    continue
                mean = float(match.group("mean"))
                std = float(match.group("std")) if match.group("std") else np.nan
            records.append(
                {
                    "paper": ARXIV_ID,
                    "source": source,
                    "metric": metric,
                    "family": current_family,
                    "case": case,
                    "method": method,
                    "mean": mean,
                    "std": std,
                    "n_runs": 3,
                }
            )
    return pd.DataFrame.from_records(records)


def extract_pinnacle_tables(tex: str) -> pd.DataFrame:
    """Extract the main L2RE claim table and detailed appendix metrics."""
    tables = [
        _parse_table(_table_before(tex, r"\label{tb1}"), "main_table"),
        _parse_table(_table_before(tex, r"\label{main-l2re}"), "appendix_table"),
        _parse_table(_table_before(tex, r"\label{main-l1re}"), "appendix_table"),
        # The max-error table accidentally repeats the main-l2re label in source.
        _parse_table(_table_before(tex, r"\label{main-l2re}", occurrence=2), "appendix_table"),
    ]
    return pd.concat(tables, ignore_index=True)


def extract_collocation_ablation(tex: str) -> pd.DataFrame:
    """Extract the published PINN versus PINN-LRA collocation ablation."""
    table = _table_before(tex, r"\label{ab-bsize}")
    rows = [
        re.sub(r"\s+", " ", row).strip()
        for row in re.split(r"\\\\(?:hline)?", table)
        if "&" in row
    ]
    header = next(row for row in rows if "Burgers1d" in row and "Poisson2d-C" in row)
    cases = [_clean(cell) for cell in header.split("&")][-4:]
    current_method = ""
    records: list[dict[str, object]] = []
    for row in rows:
        cells = [_clean(cell) for cell in row.split("&")]
        if len(cells) < 6 or not re.fullmatch(r"(?:512|2048|8192|32768)", cells[1]):
            continue
        if cells[0]:
            match = re.search(r"multirow4\*([A-Za-z-]+)", cells[0])
            current_method = match.group(1) if match else cells[0]
        points = int(cells[1])
        for case, raw in zip(cases, cells[2:6]):
            match = NUMBER_RE.search(raw)
            if not match:
                continue
            records.append(
                {
                    "paper": ARXIV_ID,
                    "metric": "L2RE",
                    "case": case,
                    "method": current_method,
                    "collocation_points": points,
                    "mean": float(match.group("mean")),
                    "std": float(match.group("std")),
                    "n_runs": 3,
                }
            )
    return pd.DataFrame.from_records(records)


def extract_to_csv(output: Path, tex_path: Path | None = None) -> pd.DataFrame:
    tex = tex_path.read_text(encoding="utf-8") if tex_path else download_tex()
    frame = extract_pinnacle_tables(tex)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def extract_collocation_to_csv(output: Path, tex_path: Path | None = None) -> pd.DataFrame:
    tex = tex_path.read_text(encoding="utf-8") if tex_path else download_tex()
    frame = extract_collocation_ablation(tex)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame
