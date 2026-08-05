"""Render the results chart as SVG, light and dark.

Generated from results/*.json so the chart cannot drift from the numbers. GitHub swaps the
two files via <picture media="(prefers-color-scheme: dark)">, which is the only theme
mechanism it honours in a README.

    uv run python scripts/make_chart.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path("results")
OUT = Path("docs")

# Categorical slots 1 and 2 from the validated reference palette. Both modes pass the
# lightness band, chroma floor, CVD separation, normal-vision floor, and contrast checks.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e4e3df",
        "series": ("#2a78d6", "#eb6834"),
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#33322f",
        "series": ("#3987e5", "#d95926"),
    },
}

PANELS = [
    ("Technical standards", "NIST Special Publications, 60 documents", "nist"),
    ("Commercial contracts", "CUAD, 510 documents", "cuad"),
]

WIDTH = 780
LEFT = 176  # room for the longest strategy name
RIGHT = 52  # room for the value label at the end of a full-width bar
BAR = 13  # thin marks
GAP = 2  # surface gap between adjacent bars
GROUP = 13  # gap between strategies
X_MAX = 0.9


def read(name: str) -> dict[str, float]:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        return {}
    return {
        r["strategy"]: r["ndcg_at_10"]
        for r in json.loads(path.read_text())
        if r["scoring"] == "k=10"
    }


def bar_path(x: float, y: float, w: float, h: float, r: float = 4.0) -> str:
    """A bar with its data-end rounded and its baseline end square."""
    r = min(r, w, h / 2)
    if w <= r:
        return f"M{x},{y}h{w}v{h}h{-w}z"
    return f"M{x},{y}h{w - r}a{r},{r} 0 0 1 {r},{r}v{h - 2 * r}a{r},{r} 0 0 1 {-r},{r}h{-(w - r)}z"


def render(theme: str) -> str:
    t = THEMES[theme]
    plot_w = WIDTH - LEFT - RIGHT
    out: list[str] = []
    y = 34

    def sx(v: float) -> float:
        return LEFT + v / X_MAX * plot_w

    for title, subtitle, corpus in PANELS:
        base, rr = read(corpus), read(f"{corpus}-rerank")
        if not base:
            continue
        rows = sorted(base.items(), key=lambda kv: -kv[1])

        out.append(
            f'<text x="{LEFT}" y="{y}" fill="{t["text"]}" font-size="14.5" '
            f'font-weight="600">{title}</text>'
        )
        out.append(
            f'<text x="{LEFT}" y="{y + 17}" fill="{t["muted"]}" font-size="11.5">{subtitle}</text>'
        )
        y += 36

        top = y
        for v in (0.0, 0.2, 0.4, 0.6, 0.8):
            x = sx(v)
            out.append(
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" '
                f'y2="{top + len(rows) * (BAR * 2 + GAP + GROUP) - GROUP}" '
                f'stroke="{t["grid"]}" stroke-width="1"/>'
            )

        for strategy, score in rows:
            # Non-breaking hyphen so long strategy names never wrap in the label column.
            label = strategy.replace("-", chr(0x2011))
            out.append(
                f'<text x="{LEFT - 12}" y="{y + BAR + 4}" fill="{t["text"]}" '
                f'font-size="12" text-anchor="end">{label}</text>'
            )

            for i, (value, colour) in enumerate(
                ((score, t["series"][0]), (rr.get(strategy), t["series"][1]))
            ):
                if value is None:
                    continue
                by = y + i * (BAR + GAP)
                w = max(1.0, sx(value) - LEFT)
                out.append(f'<path d="{bar_path(LEFT, by, w, BAR)}" fill="{colour}"/>')
                out.append(
                    f'<text x="{LEFT + w + 7:.1f}" y="{by + BAR - 2.5}" '
                    f'fill="{t["muted"]}" font-size="10.5">{value:.3f}</text>'
                )

            y += BAR * 2 + GAP + GROUP

        # Tick labels, so the gridlines carry a scale rather than decorating.
        for v in (0.0, 0.2, 0.4, 0.6, 0.8):
            out.append(
                f'<text x="{sx(v):.1f}" y="{y - 2}" fill="{t["muted"]}" font-size="10" '
                f'text-anchor="middle">{v:.1f}</text>'
            )
        y += 22

    # Legend. Two series, so it is always present, and the values above are direct labels.
    ly = y + 2
    for i, name in enumerate(("without reranking", "with reranking")):
        lx = LEFT + i * 168
        out.append(
            f'<rect x="{lx}" y="{ly - 9}" width="11" height="11" rx="2.5" fill="{t["series"][i]}"/>'
        )
        out.append(
            f'<text x="{lx + 17}" y="{ly}" fill="{t["muted"]}" font-size="11.5">{name}</text>'
        )

    out.append(
        f'<text x="{LEFT}" y="{ly + 22}" fill="{t["muted"]}" font-size="10.5">'
        f"nDCG@10 at k=10. Higher is better. Each corpus scored in the query form a user "
        f"would type.</text>"
    )

    height = ly + 38
    body = "\n  ".join(out)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" font-family="-apple-system, BlinkMacSystemFont, '
        f"'Segoe UI', Helvetica, Arial, sans-serif\">\n"
        f'  <rect width="{WIDTH}" height="{height}" fill="{t["surface"]}"/>\n'
        f"  {body}\n</svg>\n"
    )


def main() -> int:
    OUT.mkdir(exist_ok=True)
    for theme in THEMES:
        path = OUT / f"results-{theme}.svg"
        path.write_text(render(theme))
        print(f"wrote {path} ({path.stat().st_size / 1024:.1f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
