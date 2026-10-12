"""CarbonPlan-themed matplotlib output for the executed docs pages."""

import base64
from collections.abc import Callable
from io import BytesIO

import carbonplan_styles as cps
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.figure import Figure


def render(plot: Callable[[str], Figure]) -> str:
    """Render plot(mode) for light and dark, as <img> tags Material shows per color scheme."""
    imgs = []
    for mode in ("light", "dark"):
        with plt.style.context(cps.style(mode)):
            fig = plot(mode)
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        data = base64.b64encode(buf.getvalue()).decode()
        imgs.append(f'<img src="data:image/png;base64,{data}#only-{mode}">')
    return "\n".join(imgs)


def html_repr(obj) -> str:
    """xarray's notebook repr in a shadow root, so Material's typeset CSS can't restyle it."""
    # attrs hold long convention dicts and WKT; start them collapsed
    with xr.set_options(display_expand_attrs=False):
        html = obj._repr_html_()
    # dask chunk diagram labels inherit a black stroke; draw them in the page text color
    style = "<style>svg text { fill: currentColor; stroke: none; }</style>"
    return f'<div><template shadowrootmode="open">{style}{html}</template></div>'
