"""
geo_agents.projection
======================
CAPABILITY 2 -- Map projection / reprojection  (deterministic)

Chooses a suitable CRS and reprojects vector datasets. Honours an explicit
EPSG/ESRI code in the request, recognises named systems (Web Mercator, WGS 84,
CONUS Albers), and otherwise estimates a local UTM zone for metric accuracy.

Standalone use:
    from geo_agents.projection import run
    run("reproject roads.gpkg to EPSG:3857", "roads.gpkg")
    run("project parcels.gpkg to UTM", "parcels.gpkg")
"""

from __future__ import annotations

import os
import re
from typing import Callable, List, Optional, Tuple

from .common import _read_vector, _preferred_vector_ext, _write_vector, make_host

__all__ = ["ProjectionCapability", "run"]


class ProjectionCapability:
    key = "map_projection"
    produces_data = True
    needs_input = True
    description = "Choose a suitable CRS and reproject vector datasets."
    keywords = ["reproject", "projection", "project to", "crs", "epsg", "srs",
                "coordinate system", "coordinate reference", "utm", "web mercator",
                "pseudo mercator", "wgs84", "wgs 84", "albers", "lambert",
                "transform crs", "change projection"]

    def __init__(self, agent):
        self.agent = agent

    def _extract_explicit_crs(self, query: str) -> Optional[str]:
        text = query or ""
        m = re.search(r"\b(EPSG|ESRI|IGNF|OGC)[:\s-]*(\d{3,6})\b", text, flags=re.IGNORECASE)
        if m:
            return f"{m.group(1).upper()}:{m.group(2)}"
        m = re.search(r"\b(?:EPSG[:\s-]*)?(\d{4,6})\b", text, flags=re.IGNORECASE)
        if not m:
            return None
        code = m.group(1)
        ctx = r"(EPSG|CRS|projection|project|reproject|coordinate|transform|srs)"
        if re.search(rf"{ctx}[^\n]{{0,80}}{code}", text, flags=re.IGNORECASE) or \
           re.search(rf"{code}[^\n]{{0,80}}{ctx}", text, flags=re.IGNORECASE):
            return f"EPSG:{code}"
        return None

    def _select_crs(self, query: str, gdf) -> Tuple[str, str]:
        from pyproj import CRS
        text = (query or "").lower()
        explicit = self._extract_explicit_crs(query)
        if explicit:
            return CRS.from_user_input(explicit).to_string(), \
                f"The request explicitly specified {explicit}."
        named = [
            (("web mercator", "pseudo mercator", "slippy", "tile map"), "EPSG:3857",
             "Web Mercator, the standard for web map tiles."),
            (("wgs84", "wgs 84", "lat lon", "latitude longitude", "longitude latitude"),
             "EPSG:4326", "WGS 84 geographic coordinates."),
            (("conus albers", "contiguous albers"), "EPSG:5070",
             "NAD83 / CONUS Albers equal-area."),
        ]
        for keys, code, reason in named:
            if any(k in text for k in keys):
                return CRS.from_user_input(code).to_string(), reason
        wants_distance = any(t in text for t in
                             ("utm", "distance", "buffer", "nearest", "length",
                              "meters", "metres", "area", "metric"))
        if wants_distance:
            try:
                est = gdf.estimate_utm_crs()
                return est.to_string(), ("Estimated a local UTM CRS from the data "
                                         "extent for metric accuracy.")
            except Exception:
                pass
        try:
            est = gdf.estimate_utm_crs()
            return est.to_string(), "No CRS specified; estimated a local UTM CRS."
        except Exception:
            return "EPSG:4326", "Fell back to WGS 84."

    def run(self, query: str, input_paths: List[str],
            progress_callback: Optional[Callable]) -> dict:
        from pyproj import CRS
        if not input_paths:
            raise ValueError("Reprojection needs an input dataset.")
        agent = self.agent
        outputs, summaries = [], []
        for path in input_paths:
            gdf = _read_vector(path)
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=4326, allow_override=True)
            target, reason = self._select_crs(query, gdf)
            agent._emit_progress(progress_callback, "projection_select",
                                 f"Reprojecting {os.path.basename(path)} to {target}.",
                                 {"target_crs": target, "reason": reason})
            transformed = gdf.to_crs(CRS.from_user_input(target))
            ext = _preferred_vector_ext(query)
            out = agent._out_path(f"reprojected {query}", ext, "reprojected")
            outputs.append(_write_vector(transformed, out))
            summaries.append(f"{os.path.basename(path)} -> {target} ({reason})")
        return {"text": "Reprojected: " + "; ".join(summaries),
                "dataset_paths": outputs}


def run(query: str, input_dataset_paths=None, *, agent=None, progress_callback=None,
        provider=None, model=None, api_key=None, output_dir=None) -> dict:
    """Run the projection capability standalone."""
    host = agent or make_host(api_key=api_key, model=model,
                              output_dir=output_dir, provider=provider)
    cap = ProjectionCapability(host)
    paths = host.normalize_dataset_paths(input_dataset_paths)
    return cap.run(query, paths, progress_callback)
