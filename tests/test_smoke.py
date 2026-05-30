"""
Smoke tests: the package and every capability must import, register, and route
WITHOUT the heavy GIS / AI dependencies installed. The actual geospatial ops are
not exercised here (they need geopandas etc. and real data); these tests guard
the structure, public API, and offline keyword routing.
"""

import importlib

import geo_agents


def test_version():
    assert isinstance(geo_agents.__version__, str)


def test_capability_modules_import_individually():
    for mod in (
        "data_discovery", "projection", "vector_analysis", "raster",
        "spatial_statistics", "mapping", "web_mapping",
    ):
        m = importlib.import_module(f"geo_agents.{mod}")
        assert hasattr(m, "run"), f"{mod} is missing a standalone run()"


def test_registry_has_seven_capabilities():
    assert len(geo_agents.CAPABILITIES) == 7


def test_unified_agent_registers_all_capabilities():
    agent = geo_agents.UnifiedGeoAgent(provider="ollama")
    assert set(agent.capabilities) == set(geo_agents.CAPABILITIES)


def test_offline_keyword_routing_works_without_llm():
    agent = geo_agents.UnifiedGeoAgent(provider="ollama")
    # No model reachable in CI -> client is None -> heuristic routing.
    route = agent._heuristic_route("download the parks of Tehran", has_datasets=False)
    assert route and route[0]["capability"] == "data_discovery"
