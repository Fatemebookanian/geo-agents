# geo-agents

A modular GIS agent suite. One text request → the right geospatial operation.
Seven independent capabilities you can import on their own, or chain together
under a single `UnifiedGeoAgent`.

## Install

The base package has no required dependencies, so it always imports. The actual
GIS work needs the `gis` extra; AI routing needs the `ai` extra.

```bash
pip install "geo-agents[all] @ git+https://github.com/Fatemebookanian/geo-agents.git"
```

## AI provider

All free except OpenAI. Never hardcode a key — use an environment variable.
Ollama is the default and needs no key (install from https://ollama.com, then
`ollama pull llama3.1`). Without a reachable model the agent still runs by
routing on keywords.

```bash
export GROQ_API_KEY=...        # provider="groq"   (free cloud)
export GEMINI_API_KEY=...      # provider="gemini" (free cloud)
```

## Usage

Use one capability on its own:

```python
from geo_agents.vector_analysis import run
run("buffer roads.gpkg by 100 meters", "roads.gpkg")
```

Or the unified agent, which auto-routes and chains capabilities:

```python
from geo_agents import UnifiedGeoAgent, ask, launch_ui

ask("download the road network of Nuremberg, Germany, "
    "reproject to UTM, buffer 50 m, and build an interactive web map")

launch_ui()   # interactive panel inside Jupyter
```

Outputs (`.gpkg`, `.geojson`, `.png`, `.html`) are written to the output dir
(default `geo_agent_outputs/`, override with `GEO_AGENT_OUTPUT_DIR`).

## Capabilities

| Module | What it does |
|---|---|
| `geo_agents.data_discovery` | Download data from OpenStreetMap / Natural Earth |
| `geo_agents.projection` | Choose a CRS and reproject |
| `geo_agents.vector_analysis` | buffer, clip, dissolve, overlay, joins, nearest, … |
| `geo_agents.raster` | rasterize, clip raster, zonal stats, info |
| `geo_agents.spatial_statistics` | Moran's I, LISA hot/cold spots |
| `geo_agents.mapping` | static PNG maps (matplotlib) |
| `geo_agents.web_mapping` | interactive Leaflet/Folium HTML maps |
| `geo_agents.orchestrator` | `UnifiedGeoAgent` — chains them all |

## Development

```bash
git clone https://github.com/Fatemebookanian/geo-agents.git
cd geo-agents
pip install -e ".[all,dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
