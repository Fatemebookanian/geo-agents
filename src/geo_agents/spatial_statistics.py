"""
geo_agents.spatial_statistics
=============================
CAPABILITY 5 -- Spatial statistics  (PySAL / esda)

Measures spatial autocorrelation of a numeric attribute:
  * Global Moran's I  -- is the variable clustered, dispersed, or random?
  * Local Moran (LISA) -- labels each feature HH / LL / HL / LH / ns
    (hot spots, cold spots, spatial outliers) at the 0.05 level.

Spatial weights are built automatically (Queen contiguity for polygons,
KNN otherwise). Produces a text report and writes a cluster layer with
``lisa_cluster`` and ``lisa_p`` fields.

Standalone use:
    from geo_agents.spatial_statistics import run
    run("hot spot analysis of income", "tracts.gpkg")
"""

from __future__ import annotations

from typing import Callable, List, Optional

from .common import _read_vector, _write_vector, _pick_value_column, make_host

__all__ = ["SpatialStatisticsCapability", "run"]


class SpatialStatisticsCapability:
    key = "spatial_statistics"
    produces_data = True   # writes a LISA-cluster layer when possible
    needs_input = True
    description = ("PySAL-based spatial statistics: global spatial autocorrelation "
                   "(Moran's I), local indicators (LISA hot/cold spots), and "
                   "Getis-Ord G. Produces a report and a cluster layer.")
    keywords = ["moran", "geary", "autocorrelation", "hot spot", "hotspot",
                "cold spot", "getis", "ord", "lisa", "cluster", "clustering",
                "spatial regression", "spatial weights", "pysal", "statistics",
                "statistical", "spatial econometrics"]

    def __init__(self, agent):
        self.agent = agent

    def run(self, query: str, input_paths: List[str],
            progress_callback: Optional[Callable]) -> dict:
        import numpy as np
        from libpysal.weights import Queen, KNN
        from esda.moran import Moran, Moran_Local
        if not input_paths:
            raise ValueError("Spatial statistics needs an input dataset.")
        agent = self.agent
        gdf = _read_vector(input_paths[0]).reset_index(drop=True)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326, allow_override=True)
        col = _pick_value_column(gdf, query)
        if not col:
            raise ValueError("No numeric column found to analyse.")
        agent._emit_progress(progress_callback, "stats_setup",
                             f"Analysing spatial autocorrelation of '{col}'.",
                             {"value_column": col, "features": len(gdf)})

        # Build spatial weights: Queen for polygons, KNN otherwise.
        geom_type = gdf.geometry.geom_type.iloc[0]
        try:
            w = Queen.from_dataframe(gdf, use_index=False) if "Polygon" in geom_type \
                else KNN.from_dataframe(gdf, k=min(8, max(1, len(gdf) - 1)))
        except Exception:
            w = KNN.from_dataframe(gdf, k=min(8, max(1, len(gdf) - 1)))
        w.transform = "r"

        y = gdf[col].astype(float).fillna(gdf[col].astype(float).mean()).values
        mi = Moran(y, w)
        lisa = Moran_Local(y, w)

        # Label LISA clusters (HH/LL/HL/LH/ns at 0.05).
        labels = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
        sig = lisa.p_sim < 0.05
        gdf["lisa_cluster"] = [labels.get(q, "ns") if s else "ns"
                               for q, s in zip(lisa.q, sig)]
        gdf["lisa_p"] = lisa.p_sim

        out = agent._out_path(f"lisa {query}", ".gpkg", "lisa_clusters")
        _write_vector(gdf, out)

        n_hot = int((gdf["lisa_cluster"] == "HH").sum())
        n_cold = int((gdf["lisa_cluster"] == "LL").sum())
        report = (f"Spatial statistics on '{col}' ({len(gdf)} features):\n"
                  f"- Global Moran's I = {mi.I:.4f} (p = {mi.p_sim:.4f})\n"
                  f"- Interpretation: "
                  f"{'clustered' if mi.p_sim < 0.05 and mi.I > 0 else 'dispersed' if mi.p_sim < 0.05 and mi.I < 0 else 'random (not significant)'}\n"
                  f"- LISA significant hot spots (HH): {n_hot}\n"
                  f"- LISA significant cold spots (LL): {n_cold}\n"
                  f"- Cluster layer saved with 'lisa_cluster' & 'lisa_p' fields.")
        return {"text": report, "dataset_paths": [out],
                "morans_i": float(mi.I), "p_value": float(mi.p_sim)}


def run(query: str, input_dataset_paths=None, *, agent=None, progress_callback=None,
        provider=None, model=None, api_key=None, output_dir=None) -> dict:
    """Run the spatial-statistics capability standalone."""
    host = agent or make_host(api_key=api_key, model=model,
                              output_dir=output_dir, provider=provider)
    cap = SpatialStatisticsCapability(host)
    paths = host.normalize_dataset_paths(input_dataset_paths)
    return cap.run(query, paths, progress_callback)
