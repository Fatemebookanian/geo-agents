"""
geo_agents.common
=================
Shared infrastructure used by every capability ("agent") in this package.

Contains:
  * small text/JSON utilities
  * the free-by-default LLM provider system (Ollama / Groq / Gemini / OpenAI)
  * ``BaseGeoAgent`` -- the lightweight host that every capability needs
    (output dir, progress events, LLM chat, dataset normalisation)
  * vector / raster IO helpers shared by all capabilities

Nothing here is GIS-operation specific; the actual operations live in the
per-capability modules (data_discovery, projection, vector_analysis, ...).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("geo_agents")

DEFAULT_OUTPUT_DIR = Path(os.environ.get("GEO_AGENT_OUTPUT_DIR", "geo_agent_outputs"))


# =============================================================================
# Small utilities
# =============================================================================
def _slugify(text: str, fallback: str = "output", max_len: int = 40) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if not text:
        text = fallback
    return text[:max_len]


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _parse_llm_json(content: str) -> Optional[dict]:
    if not content:
        return None
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


# Provider catalog. All of these speak the OpenAI-compatible chat API, so the
# same code talks to any of them -- only base_url / model / key differ.
#
#   ollama  -> 100% FREE, fully local, NO API KEY. Best for public GitHub
#              (nothing secret to commit). Install: https://ollama.com then
#              `ollama pull llama3.1`. Runs at http://localhost:11434.
#   groq    -> FREE cloud tier, very fast. Get a free key at console.groq.com
#              and put it in the GROQ_API_KEY env var (never hardcode it).
#   gemini  -> FREE tier from Google AI Studio. Key in GEMINI_API_KEY.
#   openai  -> Paid. Key in OPENAI_API_KEY.
_LLM_PROVIDERS = {
    "ollama": {"base_url": "http://localhost:11434/v1",
               "default_model": "llama3.1", "env_key": None, "needs_key": False},
    "groq":   {"base_url": "https://api.groq.com/openai/v1",
               "default_model": "llama-3.3-70b-versatile",
               "env_key": "GROQ_API_KEY", "needs_key": True},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "default_model": "gemini-2.0-flash",
               "env_key": "GEMINI_API_KEY", "needs_key": True},
    "openai": {"base_url": "https://api.openai.com/v1",
               "default_model": "gpt-4o-mini",
               "env_key": "OPENAI_API_KEY", "needs_key": True},
}


def _resolve_provider(api_key: Optional[str], provider: Optional[str]) -> Tuple[str, Optional[str]]:
    """Pick a provider + key. Priority: explicit provider > any env key > ollama.

    Override the default with the GEO_AGENT_PROVIDER env var if you like.
    """
    provider = (provider or os.environ.get("GEO_AGENT_PROVIDER") or "").lower().strip()
    if provider in _LLM_PROVIDERS:
        spec = _LLM_PROVIDERS[provider]
        key = api_key or (os.environ.get(spec["env_key"]) if spec["env_key"] else None)
        return provider, key
    # Auto-detect from whichever key is present.
    if api_key:
        return "openai", api_key
    for name in ("groq", "gemini", "openai"):
        env_key = _LLM_PROVIDERS[name]["env_key"]
        if env_key and os.environ.get(env_key):
            return name, os.environ[env_key]
    # Default: free local Ollama (no key needed).
    return "ollama", None


def _build_llm_client(api_key: Optional[str], provider: Optional[str] = None,
                      base_url: Optional[str] = None) -> Tuple[Any, Optional[str], str]:
    """Return (client, default_model, provider_name). client is None on failure."""
    name, key = _resolve_provider(api_key, provider)
    spec = _LLM_PROVIDERS[name]
    url = base_url or spec["base_url"]
    if spec["needs_key"] and not key:
        logger.warning("Provider '%s' needs a key but none was found; "
                       "falling back to keyword routing.", name)
        return None, spec["default_model"], name
    try:
        from openai import OpenAI
        # Ollama ignores the key but the SDK requires a non-empty string.
        client = OpenAI(api_key=key or "ollama", base_url=url)
        logger.info("LLM provider: %s (model default: %s)", name, spec["default_model"])
        return client, spec["default_model"], name
    except Exception as exc:
        logger.warning("Could not build LLM client for '%s' (%s); "
                       "using keyword routing.", name, exc)
        return None, spec["default_model"], name


# =============================================================================
# Minimal base class (the host every capability needs)
# =============================================================================
class BaseGeoAgent:
    """Lightweight base: dataset normalisation, progress events, output dir."""

    agent_name = "Base Geo Agent"
    agent_version = "1.0.0"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 output_dir: Optional[Path] = None, provider: Optional[str] = None,
                 base_url: Optional[str] = None):
        self.api_key = api_key
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Build a free-by-default LLM client (Ollama if nothing else is set).
        self.client, default_model, self.provider = _build_llm_client(
            api_key, provider=provider, base_url=base_url)
        # Caller's model wins; otherwise the provider's sensible default.
        self.model = model or default_model
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_calls = 0
        self.tool_calls = 0

    @staticmethod
    def normalize_dataset_paths(paths) -> List[str]:
        if paths is None:
            return []
        if isinstance(paths, str):
            return [paths] if paths else []
        return [p for p in paths if p]

    @staticmethod
    def _emit_progress(callback: Optional[Callable], stage: str = "",
                       message: str = "", data: Optional[dict] = None) -> None:
        if callback is None:
            return
        try:
            callback({"stage": stage, "message": message, "data": data or {}})
        except Exception:
            pass

    def _out_path(self, hint: str, ext: str, fallback: str = "output") -> str:
        name = f"{_slugify(hint, fallback)}_{_timestamp()}{ext}"
        return str(self.output_dir / name)

    def _llm_chat(self, system: str, user: str, purpose: str = "") -> Optional[str]:
        if self.client is None:
            return None
        try:
            resp = self.client.chat.completions.create(
                model=self.model or "llama3.1",
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0,
            )
        except Exception as exc:
            logger.warning("LLM call failed (%s): %s", purpose, exc)
            return None
        self.llm_calls += 1
        usage = getattr(resp, "usage", None)
        if usage:
            self.input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        return resp.choices[0].message.content if resp and resp.choices else None


# =============================================================================
# Vector / raster IO helpers shared by capabilities
# =============================================================================
def _read_vector(path: str):
    import geopandas as gpd
    if path.lower().endswith((".parquet", ".pq")):
        return gpd.read_parquet(path)
    return gpd.read_file(path)


def _write_vector(gdf, path: str) -> str:
    driver = "GeoJSON" if path.lower().endswith(".geojson") else "GPKG"
    if path.lower().endswith(".shp"):
        driver = "ESRI Shapefile"
    gdf.to_file(path, driver=driver)
    return path


def _preferred_vector_ext(query: str) -> str:
    q = (query or "").lower()
    if "geojson" in q and not any(t in q for t in ("geopackage", "gpkg", ".gpkg")):
        return ".geojson"
    if "shapefile" in q or ".shp" in q:
        return ".shp"
    return ".gpkg"


def _numeric_columns(gdf) -> List[str]:
    import pandas as pd
    cols = []
    for c in gdf.columns:
        if c == gdf.geometry.name:
            continue
        if pd.api.types.is_numeric_dtype(gdf[c]):
            cols.append(c)
    return cols


def _pick_value_column(gdf, query: str) -> Optional[str]:
    nums = _numeric_columns(gdf)
    if not nums:
        return None
    q = (query or "").lower()
    for c in nums:
        if c.lower() in q:
            return c
    for c in nums:
        if not re.search(r"(^id$|_id$|fid|objectid|code|zip|fips)", c.lower()):
            return c
    return nums[0]


def _parse_distance_meters(query: str) -> float:
    q = (query or "").lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(kilometers?|km|miles?|mi|feet|ft|meters?|metres?|m)\b", q)
    if not m:
        m2 = re.search(r"(\d+(?:\.\d+)?)", q)
        return float(m2.group(1)) if m2 else 1000.0
    val = float(m.group(1))
    unit = m.group(2)
    if unit in ("km", "kilometer", "kilometers"):
        return val * 1000.0
    if unit in ("mi", "mile", "miles"):
        return val * 1609.344
    if unit in ("ft", "feet"):
        return val * 0.3048
    return val


def _to_metric_crs(gdf):
    original = gdf.crs
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
        original = gdf.crs
    try:
        from pyproj import CRS
        if CRS.from_user_input(gdf.crs).is_projected:
            return gdf, original
    except Exception:
        pass
    try:
        metric = gdf.estimate_utm_crs()
        return gdf.to_crs(metric), original
    except Exception:
        return gdf.to_crs(epsg=3857), original


# =============================================================================
# Standalone-run helper used by each capability module
# =============================================================================
def make_host(api_key: Optional[str] = None, model: Optional[str] = None,
              output_dir: Optional[Path] = None, provider: Optional[str] = None,
              base_url: Optional[str] = None) -> BaseGeoAgent:
    """Build a minimal host so a single capability can run on its own.

    Each capability needs a host object for output paths, progress events and
    (optionally) LLM calls. This returns a ready ``BaseGeoAgent`` for that.
    """
    return BaseGeoAgent(api_key=api_key, model=model, output_dir=output_dir,
                        provider=provider, base_url=base_url)
