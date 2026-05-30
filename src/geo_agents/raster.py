"""
geo_agents.raster
=================
CAPABILITY 4 -- Raster & mixed raster-vector workflows

Four operations, auto-detected from the request and the input file types:
  * rasterize  -- burn a vector layer into a GeoTIFF grid
  * clip       -- clip/mask a raster by a vector outline
  * zonal      -- zonal statistics of a raster within vector zones
  * info       -- report raster metadata (CRS, size, bands, resolution, ...)

Standalone use:
    from geo_agents.raster import run
    run("zonal stats", ["dem.tif", "zones.gpkg"])
    run("rasterize parcels", "parcels.gpkg")
    run("raster info", "dem.tif")
"""

from __future__ import annotations

import json
import re
from typing import Callable, List, Optional

from .common import (
    _read_vector, _parse_distance_meters, _pick_value_column,
    _to_metric_crs, make_host,
)

__all__ = ["RasterCapability", "run"]


class RasterCapability:
    key = "raster"
    produces_data = True
    needs_input = True
    description = ("Raster & mixed raster-vector workflows: rasterize a vector layer, "
                   "clip a raster by a vector mask, zonal statistics, and raster info.")
    keywords = ["raster", "rasterize", "geotiff", "tiff", ".tif", "pixel", "cell size",
                "resolution", "dem", "elevation", "zonal", "clip raster", "mask raster",
                "band", "reclassify"]

    def __init__(self, agent):
        self.agent = agent

    @staticmethod
    def _is_raster(path: str) -> bool:
        return path.lower().endswith((".tif", ".tiff", ".img", ".vrt", ".asc", ".jp2"))

    def _detect_op(self, query: str, input_paths: List[str]) -> str:
        q = (query or "").lower()
        rasters = [p for p in input_paths if self._is_raster(p)]
        vectors = [p for p in input_paths if not self._is_raster(p)]
        if "rasterize" in q or (vectors and not rasters):
            return "rasterize"
        if any(t in q for t in ("zonal", "statistics", "stats")) and rasters and vectors:
            return "zonal"
        if any(t in q for t in ("clip", "mask")) and rasters and vectors:
            return "clip"
        if any(t in q for t in ("info", "inspect", "describe", "metadata")):
            return "info"
        if rasters and vectors:
            return "clip"
        return "info"

    def run(self, query: str, input_paths: List[str],
            progress_callback: Optional[Callable]) -> dict:
        if not input_paths:
            raise ValueError("Raster analysis needs at least one input dataset.")
        agent = self.agent
        op = self._detect_op(query, input_paths)
        rasters = [p for p in input_paths if self._is_raster(p)]
        vectors = [p for p in input_paths if not self._is_raster(p)]
        agent._emit_progress(progress_callback, "raster_op",
                             f"Running raster operation: {op}.", {"operation": op})

        if op == "rasterize":
            return self._rasterize(query, vectors[0] if vectors else input_paths[0])
        if op == "clip":
            return self._clip(query, rasters[0], vectors[0])
        if op == "zonal":
            return self._zonal(query, rasters[0], vectors[0])
        return self._info(rasters[0] if rasters else input_paths[0])

    def _rasterize(self, query: str, vector_path: str) -> dict:
        import numpy as np
        import rasterio
        from rasterio import features
        from rasterio.transform import from_bounds
        gdf = _read_vector(vector_path)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326, allow_override=True)
        res = _parse_distance_meters(query) if re.search(r"\d", query or "") else None
        work, _ = _to_metric_crs(gdf)
        if not res:
            minx, miny, maxx, maxy = work.total_bounds
            res = max((maxx - minx), (maxy - miny)) / 1000.0
        minx, miny, maxx, maxy = work.total_bounds
        width = max(1, int((maxx - minx) / res))
        height = max(1, int((maxy - miny) / res))
        transform = from_bounds(minx, miny, maxx, maxy, width, height)
        value_col = _pick_value_column(work, query)
        shapes = ((geom, (row[value_col] if value_col else 1))
                  for geom, (_, row) in zip(work.geometry, work.iterrows()))
        arr = features.rasterize(shapes, out_shape=(height, width),
                                 transform=transform, fill=0, dtype="float32")
        out = self.agent._out_path(f"rasterized {query}", ".tif", "rasterized")
        with rasterio.open(out, "w", driver="GTiff", height=height, width=width,
                           count=1, dtype="float32", crs=work.crs,
                           transform=transform, nodata=0) as dst:
            dst.write(arr, 1)
        return {"text": f"Rasterized to {width}x{height} grid (res ~{res:g} m).",
                "dataset_paths": [out]}

    def _clip(self, query: str, raster_path: str, vector_path: str) -> dict:
        import rasterio
        from rasterio.mask import mask
        gdf = _read_vector(vector_path)
        with rasterio.open(raster_path) as src:
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=4326, allow_override=True)
            gdf = gdf.to_crs(src.crs)
            geoms = [g.__geo_interface__ for g in gdf.geometry]
            out_img, out_transform = mask(src, geoms, crop=True)
            meta = src.meta.copy()
            meta.update({"height": out_img.shape[1], "width": out_img.shape[2],
                         "transform": out_transform})
        out = self.agent._out_path(f"clipped {query}", ".tif", "clipped_raster")
        with rasterio.open(out, "w", **meta) as dst:
            dst.write(out_img)
        return {"text": "Clipped raster to the vector mask.", "dataset_paths": [out]}

    def _zonal(self, query: str, raster_path: str, vector_path: str) -> dict:
        from rasterstats import zonal_stats
        import geopandas as gpd
        gdf = _read_vector(vector_path)
        import rasterio
        with rasterio.open(raster_path) as src:
            rcrs = src.crs
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326, allow_override=True)
        if rcrs is not None:
            gdf = gdf.to_crs(rcrs)
        stats = zonal_stats(gdf, raster_path, stats=["min", "max", "mean", "count", "sum"])
        for stat in ("min", "max", "mean", "count", "sum"):
            gdf[f"zs_{stat}"] = [s.get(stat) for s in stats]
        out = self.agent._out_path(f"zonalstats {query}", ".gpkg", "zonal_stats")
        from .common import _write_vector
        _write_vector(gdf, out)
        return {"text": f"Computed zonal statistics for {len(gdf)} zone(s).",
                "dataset_paths": [out]}

    def _info(self, raster_path: str) -> dict:
        import rasterio
        with rasterio.open(raster_path) as src:
            info = {"driver": src.driver, "crs": str(src.crs),
                    "size": [src.width, src.height], "bands": src.count,
                    "dtype": src.dtypes[0], "bounds": list(src.bounds),
                    "resolution": list(src.res), "nodata": src.nodata}
        return {"text": "Raster info:\n" + json.dumps(info, indent=2),
                "dataset_paths": []}


def run(query: str, input_dataset_paths=None, *, agent=None, progress_callback=None,
        provider=None, model=None, api_key=None, output_dir=None) -> dict:
    """Run the raster capability standalone."""
    host = agent or make_host(api_key=api_key, model=model,
                              output_dir=output_dir, provider=provider)
    cap = RasterCapability(host)
    paths = host.normalize_dataset_paths(input_dataset_paths)
    return cap.run(query, paths, progress_callback)
