"""
geo_agents.vector_analysis
===========================
CAPABILITY 3 -- Vector analysis  (deterministic operations)

A single capability covering common vector GIS operations: buffer, dissolve,
clip, overlay (intersection/union/difference), spatial join, attribute join,
nearest distance, point-in-polygon counts, centroid, convex hull, simplify,
geometry repair, area/length measurement, and format conversion. The operation
is auto-detected from the request text.

Operations that need two layers (clip, overlay, joins, nearest, point counts)
expect two input paths: the primary first, the secondary second.

Standalone use:
    from geo_agents.vector_analysis import run
    run("buffer roads.gpkg by 100 meters", "roads.gpkg")
    run("clip a by b", ["a.gpkg", "b.gpkg"])
"""

from __future__ import annotations

from typing import Callable, List, Optional

from .common import (
    _read_vector, _write_vector, _preferred_vector_ext,
    _parse_distance_meters, _to_metric_crs, make_host,
)

__all__ = ["VectorAnalysisCapability", "run"]


class VectorAnalysisCapability:
    key = "vector_analysis"
    produces_data = True
    needs_input = True
    description = ("Vector GIS analysis & transformation: buffer, dissolve, clip, "
                   "overlay/intersection/union, centroid, convex hull, simplify, "
                   "spatial join, attribute join, nearest distance, "
                   "point-in-polygon counts, format conversion, geometry repair, "
                   "add area/length measurements.")
    keywords = ["buffer", "dissolve", "clip", "overlay", "intersect", "intersection",
                "union", "centroid", "convex hull", "simplify", "spatial join",
                "attribute join", "join", "nearest", "distance", "within",
                "point in polygon", "count points", "merge", "repair", "fix geometry",
                "convert to geojson", "convert to shapefile", "convert to gpkg",
                "area", "length", "vector"]

    def __init__(self, agent):
        self.agent = agent

    def _detect_op(self, query: str) -> str:
        q = (query or "").lower()
        table = [
            ("buffer", ["buffer"]),
            ("dissolve", ["dissolve", "merge by", "aggregate"]),
            ("clip", ["clip", "cookie cut", "mask to"]),
            ("overlay", ["overlay", "intersect", "intersection", "union", "difference"]),
            ("point_counts", ["count points", "point in polygon", "points in polygon",
                              "points per", "count by polygon", "points within polygon"]),
            ("spatial_join", ["spatial join", "join by location"]),
            ("attribute_join", ["attribute join", "join table", "join on", "merge attribute"]),
            ("nearest", ["nearest", "closest", "distance to"]),
            ("centroid", ["centroid", "center point", "centre point"]),
            ("convex_hull", ["convex hull", "hull"]),
            ("simplify", ["simplify", "generalize", "generalise"]),
            ("repair", ["repair", "fix geometry", "make valid", "clean geometry"]),
            ("measure", ["add area", "add length", "compute area", "compute length",
                         "calculate area", "calculate length", "measurements"]),
            ("convert", ["convert to", "export as", "save as", "to geojson",
                         "to shapefile", "to gpkg"]),
        ]
        for op, kws in table:
            if any(k in q for k in kws):
                return op
        return "buffer" if "buffer" in q else "convert"

    def run(self, query: str, input_paths: List[str],
            progress_callback: Optional[Callable]) -> dict:
        import geopandas as gpd
        import pandas as pd
        if not input_paths:
            raise ValueError("Vector analysis needs at least one input dataset.")
        agent = self.agent
        op = self._detect_op(query)
        agent._emit_progress(progress_callback, "vector_op",
                             f"Running vector operation: {op}.", {"operation": op})

        primary = _read_vector(input_paths[0])
        if primary.crs is None:
            primary = primary.set_crs(epsg=4326, allow_override=True)
        secondary = _read_vector(input_paths[1]) if len(input_paths) > 1 else None
        if secondary is not None and secondary.crs is None:
            secondary = secondary.set_crs(epsg=4326, allow_override=True)
        if secondary is not None and secondary.crs != primary.crs:
            secondary = secondary.to_crs(primary.crs)

        ext = _preferred_vector_ext(query)
        result_gdf = None
        note = ""

        if op == "buffer":
            dist = _parse_distance_meters(query)
            work, original = _to_metric_crs(primary)
            work = work.copy()
            work["geometry"] = work.geometry.buffer(dist)
            result_gdf = work.to_crs(original)
            note = f"Buffered by {dist:g} m."

        elif op == "dissolve":
            by = None
            for c in primary.columns:
                if c != primary.geometry.name and c.lower() in (query or "").lower():
                    by = c
                    break
            result_gdf = primary.dissolve(by=by) if by else \
                gpd.GeoDataFrame(geometry=[primary.unary_union], crs=primary.crs)
            note = f"Dissolved{' by ' + by if by else ' all features'}."

        elif op == "clip":
            if secondary is None:
                raise ValueError("Clip needs a second dataset (the clip mask).")
            result_gdf = gpd.clip(primary, secondary)
            note = "Clipped primary by the mask dataset."

        elif op == "overlay":
            if secondary is None:
                raise ValueError("Overlay needs a second dataset.")
            how = "intersection"
            for h in ("union", "difference", "symmetric_difference", "identity", "intersection"):
                if h.replace("_", " ") in (query or "").lower():
                    how = h
                    break
            result_gdf = gpd.overlay(primary, secondary, how=how)
            note = f"Overlay ({how})."

        elif op == "spatial_join":
            if secondary is None:
                raise ValueError("Spatial join needs a second dataset.")
            predicate = "intersects"
            for p in ("within", "contains", "intersects"):
                if p in (query or "").lower():
                    predicate = p
                    break
            result_gdf = gpd.sjoin(primary, secondary, predicate=predicate, how="left")
            note = f"Spatial join (predicate={predicate})."

        elif op == "attribute_join":
            if secondary is None:
                raise ValueError("Attribute join needs a second dataset.")
            common = [c for c in primary.columns if c in secondary.columns
                      and c != primary.geometry.name]
            if not common:
                raise ValueError("No common column found for the attribute join.")
            key = common[0]
            joined = primary.merge(
                pd.DataFrame(secondary.drop(columns=secondary.geometry.name)),
                on=key, how="left")
            result_gdf = gpd.GeoDataFrame(joined, geometry=primary.geometry.name,
                                          crs=primary.crs)
            note = f"Attribute join on '{key}'."

        elif op == "nearest":
            if secondary is None:
                raise ValueError("Nearest needs a second dataset.")
            a, original = _to_metric_crs(primary)
            b = secondary.to_crs(a.crs)
            joined = gpd.sjoin_nearest(a, b, distance_col="nearest_dist_m", how="left")
            result_gdf = joined.to_crs(original)
            note = "Computed nearest feature + distance (m)."

        elif op == "point_counts":
            if secondary is None:
                raise ValueError("Point-in-polygon counts need points + polygons.")
            polys, pts = (primary, secondary)
            if primary.geometry.geom_type.isin(["Point", "MultiPoint"]).all():
                pts, polys = primary, secondary
            joined = gpd.sjoin(pts, polys, predicate="within", how="inner")
            idx_col = "index_right" if "index_right" in joined else joined.columns[-1]
            counts = joined.groupby(idx_col).size().rename("point_count")
            result_gdf = polys.join(counts).fillna({"point_count": 0})
            note = "Counted points within each polygon."

        elif op == "centroid":
            work, original = _to_metric_crs(primary)
            work = work.copy()
            work["geometry"] = work.geometry.centroid
            result_gdf = work.to_crs(original)
            note = "Computed centroids."

        elif op == "convex_hull":
            result_gdf = primary.copy()
            result_gdf["geometry"] = primary.geometry.convex_hull
            note = "Computed convex hulls."

        elif op == "simplify":
            tol = _parse_distance_meters(query)
            work, original = _to_metric_crs(primary)
            work = work.copy()
            work["geometry"] = work.geometry.simplify(tol, preserve_topology=True)
            result_gdf = work.to_crs(original)
            note = f"Simplified with tolerance {tol:g} m."

        elif op == "repair":
            result_gdf = primary.copy()
            result_gdf["geometry"] = result_gdf.geometry.make_valid() \
                if hasattr(result_gdf.geometry, "make_valid") else result_gdf.buffer(0)
            note = "Repaired invalid geometries."

        elif op == "measure":
            work, original = _to_metric_crs(primary)
            result_gdf = primary.copy()
            gt = primary.geometry.geom_type.iloc[0] if len(primary) else ""
            if "Polygon" in gt:
                result_gdf["area_m2"] = work.geometry.area.values
            result_gdf["length_m"] = work.geometry.length.values
            note = "Added area/length measurements (m)."

        else:  # convert
            result_gdf = primary
            note = f"Converted to {ext.lstrip('.')}."

        out = agent._out_path(f"{op} {query}", ext, op)
        _write_vector(result_gdf, out)
        return {"text": f"{note} Result has {len(result_gdf)} feature(s).",
                "dataset_paths": [out], "feature_count": len(result_gdf)}


def run(query: str, input_dataset_paths=None, *, agent=None, progress_callback=None,
        provider=None, model=None, api_key=None, output_dir=None) -> dict:
    """Run the vector-analysis capability standalone."""
    host = agent or make_host(api_key=api_key, model=model,
                              output_dir=output_dir, provider=provider)
    cap = VectorAnalysisCapability(host)
    paths = host.normalize_dataset_paths(input_dataset_paths)
    return cap.run(query, paths, progress_callback)
