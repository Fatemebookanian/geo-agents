"""
geo_agents.mapping
==================
CAPABILITY 6 -- Static maps & charts  (matplotlib)

Renders a static PNG map from a prepared dataset: a choropleth/thematic map when
a numeric column is available (auto-picked, with an optional classification
scheme such as quantiles / equal_interval / natural_breaks / fisher_jenks), or a
plain geometry map otherwise.

Standalone use:
    from geo_agents.mapping import run
    run("choropleth of population", "tracts.gpkg")
    run("map natural_breaks", "regions.gpkg")
"""

from __future__ import annotations

from typing import Callable, List, Optional

from .common import _read_vector, _pick_value_column, make_host

__all__ = ["MappingCapability", "run"]


class MappingCapability:
    key = "mapping"
    produces_data = True   # produces a PNG artifact
    needs_input = True
    description = ("Generate STATIC maps (PNG): choropleth / thematic maps and "
                   "categorical maps from a prepared dataset.")
    keywords = ["map", "static map", "choropleth", "thematic", "plot", "figure",
                "png", "visualize", "visualise", "visualization", "legend",
                "symbology", "classification", "chart"]

    def __init__(self, agent):
        self.agent = agent

    def run(self, query: str, input_paths: List[str],
            progress_callback: Optional[Callable]) -> dict:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not input_paths:
            raise ValueError("Mapping needs an input dataset.")
        agent = self.agent
        gdf = _read_vector(input_paths[0])
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326, allow_override=True)
        col = _pick_value_column(gdf, query)
        agent._emit_progress(progress_callback, "map_render",
                             f"Rendering a static map"
                             + (f" of '{col}'." if col else "."),
                             {"value_column": col})

        scheme = None
        for s in ("quantiles", "equal_interval", "natural_breaks", "fisher_jenks"):
            if s.replace("_", " ") in (query or "").lower() or s in (query or "").lower():
                scheme = s
                break

        fig, ax = plt.subplots(figsize=(11, 9))
        kwargs = dict(ax=ax, edgecolor="white", linewidth=0.2)
        if col:
            kwargs.update(column=col, legend=True, cmap="viridis")
            try:
                gdf.plot(scheme=scheme or "quantiles", k=5, **kwargs)
            except Exception:
                gdf.plot(**kwargs)
            title = f"{col}"
        else:
            gdf.plot(ax=ax, color="#3b7dd8", edgecolor="white", linewidth=0.3)
            title = "Map"
        ax.set_title(title, fontsize=14)
        ax.set_axis_off()
        out = agent._out_path(f"map {query}", ".png", "map")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return {"text": f"Saved static map ({'choropleth of ' + col if col else 'geometry'}).",
                "dataset_paths": [out]}


def run(query: str, input_dataset_paths=None, *, agent=None, progress_callback=None,
        provider=None, model=None, api_key=None, output_dir=None) -> dict:
    """Run the static-mapping capability standalone."""
    host = agent or make_host(api_key=api_key, model=model,
                              output_dir=output_dir, provider=provider)
    cap = MappingCapability(host)
    paths = host.normalize_dataset_paths(input_dataset_paths)
    return cap.run(query, paths, progress_callback)
