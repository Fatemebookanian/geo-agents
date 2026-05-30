"""
geo_agents.web_mapping
======================
CAPABILITY 7 -- Interactive web maps  (Folium / Leaflet)

Builds a browser-ready interactive HTML map from a dataset. Reprojects to
WGS 84 for Leaflet, auto-fits to the data extent, and renders either a
choropleth (when a numeric column + polygons are present) or styled features
with popups/tooltips.

Standalone use:
    from geo_agents.web_mapping import run
    run("interactive web map", "parks.gpkg")
"""

from __future__ import annotations

from typing import Callable, List, Optional

from .common import _read_vector, _pick_value_column, make_host

__all__ = ["WebMappingCapability", "run"]


class WebMappingCapability:
    key = "web_mapping"
    produces_data = True   # produces an HTML artifact
    needs_input = True
    description = ("Generate an INTERACTIVE, browser-ready web map (Folium/Leaflet "
                   "HTML) from a dataset, with popups and an optional choropleth.")
    keywords = ["web map", "web mapping", "interactive map", "leaflet", "html map",
                "browser map", "online map", "webmap", "interactive", "dashboard map"]

    def __init__(self, agent):
        self.agent = agent

    def run(self, query: str, input_paths: List[str],
            progress_callback: Optional[Callable]) -> dict:
        import folium
        if not input_paths:
            raise ValueError("Web mapping needs an input dataset.")
        agent = self.agent
        gdf = _read_vector(input_paths[0])
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326, allow_override=True)
        gdf = gdf.to_crs(epsg=4326)  # Leaflet expects lon/lat
        agent._emit_progress(progress_callback, "webmap_render",
                             "Building an interactive Leaflet map.", {})

        minx, miny, maxx, maxy = gdf.total_bounds
        center = [(miny + maxy) / 2, (minx + maxx) / 2]
        m = folium.Map(location=center, zoom_start=11, tiles="OpenStreetMap")

        col = _pick_value_column(gdf, query)
        if col and gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).any():
            gdf = gdf.reset_index(drop=True)
            gdf["_id"] = gdf.index.astype(str)
            try:
                folium.Choropleth(
                    geo_data=gdf.to_json(), data=gdf, columns=["_id", col],
                    key_on="feature.properties._id", fill_color="YlOrRd",
                    legend_name=col, line_weight=0.3,
                ).add_to(m)
            except Exception:
                folium.GeoJson(gdf.to_json()).add_to(m)
        else:
            popup_fields = [c for c in gdf.columns if c != gdf.geometry.name][:5]
            folium.GeoJson(
                gdf.to_json(),
                tooltip=folium.GeoJsonTooltip(fields=popup_fields) if popup_fields else None,
                marker=folium.CircleMarker(radius=4, fill=True),
            ).add_to(m)

        try:
            m.fit_bounds([[miny, minx], [maxy, maxx]])
        except Exception:
            pass
        out = agent._out_path(f"webmap {query}", ".html", "webmap")
        m.save(out)
        return {"text": f"Saved interactive web map ({len(gdf)} feature(s)).",
                "dataset_paths": [out]}


def run(query: str, input_dataset_paths=None, *, agent=None, progress_callback=None,
        provider=None, model=None, api_key=None, output_dir=None) -> dict:
    """Run the web-mapping capability standalone."""
    host = agent or make_host(api_key=api_key, model=model,
                              output_dir=output_dir, provider=provider)
    cap = WebMappingCapability(host)
    paths = host.normalize_dataset_paths(input_dataset_paths)
    return cap.run(query, paths, progress_callback)
