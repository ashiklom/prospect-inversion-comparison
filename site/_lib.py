"""Shared helpers for the website: paths, data loading, palette, Plotly theme."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path.cwd()
if not (ROOT / "config").exists():  # rendered with execute-dir: project (site/)
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "python"))
SUMMARY = ROOT / "results" / "summary"

# ---- Identity: one fixed color + marker per backend (validated categorical
# palette, adjacent order; markers are the secondary encoding for scatters).
BACKENDS = ["stan", "pymc", "turing", "nimble", "bayestools"]
LABEL = {"stan": "Stan", "pymc": "PyMC", "turing": "Turing.jl", "nimble": "NIMBLE",
         "bayestools": "BayesianTools"}
COLOR = {"stan": "#2a78d6", "pymc": "#eb6834", "turing": "#1baf7a", "nimble": "#eda100",
         "bayestools": "#e87ba4"}
SYMBOL = {"stan": "circle", "pymc": "square", "turing": "diamond", "nimble": "triangle-up",
          "bayestools": "x"}
ERRMODEL_LABEL = {"iid": "iid normal", "iid-approx": "iid normal, approx. E1",
                  "hetero_ar1": "heteroskedastic + AR(1)"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BAND = "rgba(42,120,214,0.18)"  # slot-1 blue at low alpha for intervals

TRAIT_UNITS = {"Cab": "µg cm⁻²", "Car": "µg cm⁻²", "Canth": "µg cm⁻²", "Cbrown": "–",
               "Cw": "g cm⁻² (EWT)", "Cm": "g cm⁻² (LMA)", "N": "–", "sigma": "–"}
TRAIT_NAME = {"Cab": "Chlorophyll a+b", "Car": "Carotenoids", "Canth": "Anthocyanins",
              "Cbrown": "Brown pigments", "Cw": "Water (EWT)", "Cm": "Dry matter (LMA)",
              "N": "Structure N", "sigma": "Residual SD σ"}

pio.templates["prospect"] = go.layout.Template(layout=dict(
    font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK, size=13),
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    xaxis=dict(exponentformat="e", gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=INK2),
               title=dict(font=dict(color=INK2))),
    yaxis=dict(exponentformat="e", gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=INK2),
               title=dict(font=dict(color=INK2))),
    legend=dict(font=dict(color=INK2), bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor="white", font=dict(color=INK)),
    margin=dict(l=60, r=20, t=50, b=50),
    hovermode="closest",
))
pio.templates.default = "prospect"


def show(fig, height=None):
    """Display a Plotly figure that uses the site's local plotly.min.js
    (loaded once per page via _quarto.yml), so the site works offline."""
    from IPython.display import HTML, display
    if height:
        fig.update_layout(height=height)
    display(HTML(fig.to_html(full_html=False, include_plotlyjs=False,
                             config={"displaylogo": False, "responsive": True})))


try:  # itables: bundle its JS in the page (offline)
    from itables import init_notebook_mode
    init_notebook_mode(all_interactive=False, connected=False)
except Exception:
    pass


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(SUMMARY / name)


def config(name: str):
    return json.load(open(ROOT / "config" / f"{name}.json"))


def traits() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "validation" / "traits.csv")


def sample_order() -> list[str]:
    return [s["id"] for s in config("validation")["samples"]]


def sample_label(sid: str) -> str:
    t = traits().set_index("sample_id")
    if sid not in t.index or sid == "synthetic":
        return "synthetic"
    sp = str(t.loc[sid, "species"]).split()
    return f"{sid} ({sp[0]} {sp[1] if len(sp) > 1 else ''})".strip()


def fmt_table(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    out = df.copy()
    for c in out.select_dtypes("number").columns:
        name = str(c).lower()
        if "rhat" in name or "r̂" in name:
            f = lambda v: f"{v:.3f}"  # noqa: E731
        elif name in ("n", "fits", "n_chains", "n_draws", "divergences") or name.startswith("fits with"):
            f = lambda v: f"{v:.0f}"  # noqa: E731
        else:
            f = lambda v: f"{v:.{digits}g}"  # noqa: E731
        out[c] = out[c].map(lambda v: "" if pd.isna(v) else f(v))
    return out


def show_table(df: pd.DataFrame, **kwargs):
    from itables import show
    show(df.reset_index(drop=True), classes="display compact", paging=len(df) > 15, showIndex=False,
         scrollX=True, **kwargs)
