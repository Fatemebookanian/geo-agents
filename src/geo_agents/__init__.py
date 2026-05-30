"""
geo_agents
==========
A modular GIS agent suite. Each capability ("agent") lives in its own module
and can be imported and used on its own:

    geo_agents.data_discovery       download from OpenStreetMap / Natural Earth
    geo_agents.projection           choose a CRS and reproject
    geo_agents.vector_analysis      buffer, clip, dissolve, overlay, joins, ...
    geo_agents.raster               rasterize, clip raster, zonal stats, info
    geo_agents.spatial_statistics   Moran's I, LISA hot/cold spots
    geo_agents.mapping              static PNG maps (matplotlib)
    geo_agents.web_mapping          interactive Leaflet/Folium HTML maps
    geo_agents.orchestrator         the UnifiedGeoAgent that chains them all

Use one capability on its own
-----------------------------
    from geo_agents.vector_analysis import run
    run("buffer roads.gpkg by 100 meters", "roads.gpkg")

    # or use the class directly with your own host object:
    from geo_agents.common import make_host
    from geo_agents.web_mapping import WebMappingCapability
    host = make_host(output_dir="out")
    WebMappingCapability(host).run("web map", ["parks.gpkg"], None)

Use the unified agent (auto-routes + chains)
--------------------------------------------
    from geo_agents import UnifiedGeoAgent, ask, launch_ui
    ask("download the parks of Tehran, Iran and build an interactive web map")
"""

from __future__ import annotations

# Shared infrastructure.
from .common import (
    BaseGeoAgent, make_host, DEFAULT_OUTPUT_DIR,
)

# Each capability class.
from .data_discovery import DataDiscoveryCapability
from .projection import ProjectionCapability
from .vector_analysis import VectorAnalysisCapability
from .raster import RasterCapability
from .spatial_statistics import SpatialStatisticsCapability
from .mapping import MappingCapability
from .web_mapping import WebMappingCapability

# The orchestrator + notebook helpers.
from .orchestrator import (
    UnifiedGeoAgent, GeoSuiteAgent, get_agent, ask, launch_ui,
)

__version__ = "2.1.0"

# A simple name -> capability-class registry for discovery/introspection.
CAPABILITIES = {
    "data_discovery": DataDiscoveryCapability,
    "map_projection": ProjectionCapability,
    "vector_analysis": VectorAnalysisCapability,
    "raster": RasterCapability,
    "spatial_statistics": SpatialStatisticsCapability,
    "mapping": MappingCapability,
    "web_mapping": WebMappingCapability,
}

__all__ = [
    "BaseGeoAgent", "make_host", "DEFAULT_OUTPUT_DIR",
    "DataDiscoveryCapability", "ProjectionCapability", "VectorAnalysisCapability",
    "RasterCapability", "SpatialStatisticsCapability", "MappingCapability",
    "WebMappingCapability",
    "UnifiedGeoAgent", "GeoSuiteAgent", "get_agent", "ask", "launch_ui",
    "CAPABILITIES",
]
