"""
geo_agents.data_discovery
==========================
CAPABILITY 1 -- Data discovery / download  (GLOBAL, open-source)

Downloads GIS data by place name + feature type from global open sources:
OpenStreetMap (via OSMnx / Overpass) and Natural Earth (public-domain admin
boundaries). Use this when the user needs to FIND or DOWNLOAD data they do not
already have.

Standalone use:
    from geo_agents.data_discovery import run
    run("download the parks of Tehran, Iran")
    run("restaurants in Nuremberg, Germany", provider="groq")
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .common import (
    _parse_llm_json, _preferred_vector_ext, _write_vector, make_host, logger,
)

__all__ = ["DataDiscoveryCapability", "run"]


class DataDiscoveryCapability:
    """Download global open geospatial data by place name + feature type."""

    key = "data_discovery"
    produces_data = True
    needs_input = False
    description = ("Discover and download GIS data from global open sources "
                   "(OpenStreetMap via OSMnx, and Natural Earth for country/admin "
                   "boundaries). Use when the user needs to FIND or DOWNLOAD data "
                   "they do not already have.")
    keywords = [
        # --- افعالِ درخواست داده ---
        "download", "discover", "find data", "fetch", "get data", "obtain",
        "acquire", "search for data", "extract", "export", "pull", "load",
        # --- منابع و مرز ---
        "openstreetmap", "osm", "overpass", "natural earth",
        "boundary", "boundaries", "admin", "administrative", "outline", "extent",
        # --- جاده و حمل‌ونقل (highway/railway/aeroway) ---
        "road network", "roads", "highway", "street", "streets", "motorway",
        "footway", "cycleway", "path", "track", "pedestrian",
        "railway", "rail", "subway", "tram", "train station", "bus stop",
        "public transport", "airport", "runway",
        # --- ساختمان و سازه (building/man_made) ---
        "buildings", "building", "house", "tower", "bridge", "lighthouse",
        # --- فضای سبز و تفریح (leisure) ---
        "parks", "park", "garden", "playground", "pitch", "stadium",
        "swimming pool", "leisure",
        # --- طبیعت و آب (natural/waterway) ---
        "natural", "water", "wood", "forest", "tree", "peak", "beach", "wetland",
        "rivers", "river", "stream", "canal", "lake", "waterway", "coastline",
        # --- کاربری زمین (landuse) ---
        "landuse", "residential area", "commercial", "industrial", "farmland",
        "meadow", "grass", "retail", "cemetery", "orchard", "vineyard",
        # --- خدمات و امکانات (amenity/healthcare) ---
        "amenities", "amenity", "restaurant", "restaurants", "cafe", "bar",
        "fast food", "school", "schools", "university", "college", "kindergarten",
        "hospital", "hospitals", "clinic", "pharmacy", "dentist",
        "bank", "atm", "post office", "police", "fire station", "library",
        "fuel", "gas station", "parking", "toilets", "drinking water",
        "place of worship", "mosque", "church", "fountain", "bench",
        # --- خرید (shop) ---
        "shop", "shops", "supermarket", "bakery", "convenience", "mall",
        "clothes", "market",
        # --- گردشگری و تاریخ (tourism/historic) ---
        "tourism", "landmark", "hotel", "hostel", "guest house", "museum",
        "attraction", "viewpoint", "artwork", "information",
        "historic", "castle", "monument", "memorial", "ruins", "archaeological site",
        # --- زیرساخت (power/barrier/office) ---
        "power", "power line", "substation", "barrier", "fence", "wall",
        "gate", "office",
        # --- اصطلاحات فارسیِ درخواست (فقط برای تشخیص قصد کاربر) ---
        "دانلود", "دریافت", "بگیر", "پیدا کن", "جستجو", "استخراج",
        "داده", "اطلاعات", "نقشه", "عوارض", "لایه",
        "مرز", "محدوده", "مرزها", "مرز اداری",
        "جاده", "خیابان", "بزرگراه", "راه", "راه‌آهن", "مترو", "ایستگاه", "فرودگاه",
        "ساختمان", "خانه", "برج", "پل",
        "پارک", "باغ", "زمین بازی", "ورزشگاه", "استخر",
        "آب", "رودخانه", "نهر", "دریاچه", "جنگل", "ساحل", "درخت",
        "کاربری زمین", "مسکونی", "تجاری", "صنعتی", "کشاورزی", "قبرستان",
        "امکانات", "خدمات", "رستوران", "کافه", "مدرسه", "دانشگاه",
        "بیمارستان", "داروخانه", "درمانگاه", "بانک", "خودپرداز",
        "پلیس", "آتش‌نشانی", "کتابخانه", "پارکینگ", "پمپ بنزین",
        "مسجد", "کلیسا", "نیایشگاه", "آبنما", "نیمکت",
        "فروشگاه", "سوپرمارکت", "نانوایی", "بازار",
        "هتل", "موزه", "جاذبه", "گردشگری", "تاریخی", "قلعه", "بنای یادبود",
        "برق", "دکل", "نرده", "دیوار", "دفتر",
    ]

    # Map common words in the request to OSM tag filters.
    _OSM_TAGS = {
        # highway
        "road": {"highway": True}, "roads": {"highway": True},
        "street": {"highway": True}, "highway": {"highway": True},
        "motorway": {"highway": "motorway"}, "footway": {"highway": "footway"},
        "path": {"highway": "path"}, "cycleway": {"highway": "cycleway"},
        "pedestrian": {"highway": "pedestrian"}, "bus_stop": {"highway": "bus_stop"},
        # railway / aeroway
        "railway": {"railway": True}, "rail": {"railway": True},
        "subway": {"railway": "subway"}, "tram": {"railway": "tram"},
        "station": {"railway": "station"}, "airport": {"aeroway": "aerodrome"},
        "runway": {"aeroway": "runway"},
        # building / man_made
        "building": {"building": True}, "buildings": {"building": True},
        "tower": {"man_made": "tower"}, "bridge": {"man_made": "bridge"},
        "lighthouse": {"man_made": "lighthouse"},
        # leisure
        "park": {"leisure": "park"}, "parks": {"leisure": "park"},
        "garden": {"leisure": "garden"}, "playground": {"leisure": "playground"},
        "pitch": {"leisure": "pitch"}, "stadium": {"leisure": "stadium"},
        "swimming_pool": {"leisure": "swimming_pool"},
        # natural / waterway
        "water": {"natural": "water"}, "wood": {"natural": "wood"},
        "tree": {"natural": "tree"}, "beach": {"natural": "beach"},
        "peak": {"natural": "peak"}, "wetland": {"natural": "wetland"},
        "river": {"waterway": "river"}, "rivers": {"waterway": "river"},
        "stream": {"waterway": "stream"}, "canal": {"waterway": "canal"},
        "waterway": {"waterway": True},
        # landuse
        "forest": {"landuse": "forest"}, "landuse": {"landuse": True},
        "farmland": {"landuse": "farmland"}, "residential": {"landuse": "residential"},
        "commercial": {"landuse": "commercial"}, "industrial": {"landuse": "industrial"},
        "meadow": {"landuse": "meadow"}, "cemetery": {"landuse": "cemetery"},
        # amenity
        "amenity": {"amenity": True}, "amenities": {"amenity": True},
        "restaurant": {"amenity": "restaurant"}, "restaurants": {"amenity": "restaurant"},
        "cafe": {"amenity": "cafe"}, "bar": {"amenity": "bar"},
        "fast_food": {"amenity": "fast_food"},
        "school": {"amenity": "school"}, "schools": {"amenity": "school"},
        "university": {"amenity": "university"}, "college": {"amenity": "college"},
        "hospital": {"amenity": "hospital"}, "hospitals": {"amenity": "hospital"},
        "clinic": {"amenity": "clinic"}, "pharmacy": {"amenity": "pharmacy"},
        "bank": {"amenity": "bank"}, "atm": {"amenity": "atm"},
        "police": {"amenity": "police"}, "fire_station": {"amenity": "fire_station"},
        "library": {"amenity": "library"}, "fuel": {"amenity": "fuel"},
        "parking": {"amenity": "parking"}, "fountain": {"amenity": "fountain"},
        "place_of_worship": {"amenity": "place_of_worship"},
        "mosque": {"amenity": "place_of_worship"},
        # shop
        "shop": {"shop": True}, "shops": {"shop": True},
        "supermarket": {"shop": "supermarket"}, "bakery": {"shop": "bakery"},
        "mall": {"shop": "mall"},
        # tourism / historic
        "tourism": {"tourism": True}, "landmark": {"tourism": True},
        "hotel": {"tourism": "hotel"}, "museum": {"tourism": "museum"},
        "attraction": {"tourism": "attraction"}, "viewpoint": {"tourism": "viewpoint"},
        "historic": {"historic": True}, "castle": {"historic": "castle"},
        "monument": {"historic": "monument"}, "ruins": {"historic": "ruins"},
        # power / barrier / office / boundary
        "power": {"power": True}, "substation": {"power": "substation"},
        "barrier": {"barrier": True}, "fence": {"barrier": "fence"},
        "wall": {"barrier": "wall"}, "gate": {"barrier": "gate"},
        "office": {"office": True}, "boundary": {"boundary": "administrative"},
    }

    def __init__(self, agent):
        self.agent = agent

    # ---- request parsing --------------------------------------------------
    def _parse_request(self, query: str) -> Dict[str, Any]:
        """Determine place, feature type, and OSM tags from the request.

        Uses the LLM when available; always validates/falls back to regex so it
        works offline too.
        """
        parsed = {"place": None, "feature": None, "tags": None,
                  "source": "osm", "admin_only": False}

        # LLM-assisted extraction (optional)
        raw = self.agent._llm_chat(
            system="You extract structured parameters for a global GIS data "
                   "download from OpenStreetMap. Return ONLY JSON.",
            user=json.dumps({
                "request": query,
                "return_schema": {
                    "place": "the place/area name to download for, e.g. 'Nuremberg, Germany'",
                    "feature": "one of: roads, buildings, parks, water, rivers, "
                               "railway, amenities, restaurants, schools, hospitals, "
                               "landuse, forest, boundary, or null for a whole-area boundary",
                    "admin_boundary_only": "true if the user only wants the area "
                                           "boundary/outline itself",
                },
                "instructions": "Infer the place from the request. If only a "
                                "boundary/outline is requested, set admin_boundary_only true.",
            }),
            purpose="data_discovery_parse",
        )
        payload = _parse_llm_json(raw) if raw else None
        if payload:
            parsed["place"] = payload.get("place") or parsed["place"]
            feat = (payload.get("feature") or "").strip().lower() or None
            parsed["feature"] = feat
            parsed["admin_only"] = bool(payload.get("admin_boundary_only"))

        # Regex fallback / validation for the place.
        if not parsed["place"]:
            m = re.search(r"\b(?:of|in|for|around|near)\s+([A-Z][\w'.-]+(?:[ ,]+[A-Z][\w'.-]+){0,4})",
                          query or "")
            if m:
                parsed["place"] = m.group(1).strip(" ,.")
        # Feature fallback by keyword.
        if not parsed["feature"]:
            ql = (query or "").lower()
            for word, tags in self._OSM_TAGS.items():
                if re.search(rf"\b{re.escape(word)}\b", ql):
                    parsed["feature"] = word
                    break

        if parsed["feature"] in (None, "boundary") or parsed["admin_only"]:
            parsed["admin_only"] = parsed["admin_only"] or parsed["feature"] in (None, "boundary")
        if parsed["feature"] and parsed["feature"] in self._OSM_TAGS:
            parsed["tags"] = self._OSM_TAGS[parsed["feature"]]
        return parsed

    # ---- download backends ------------------------------------------------
    def _download_osm(self, place: str, feature: Optional[str], tags: Optional[dict],
                      admin_only: bool, query: str) -> Tuple[Any, str]:
        import osmnx as ox
        if admin_only or not tags:
            # Area outline / administrative boundary.
            if feature == "roads" or (feature is None and "road" in (query or "").lower()):
                graph = ox.graph_from_place(place, network_type="drive")
                gdf = ox.graph_to_gdfs(graph, nodes=False)
                return gdf.reset_index(), f"OSM road network for {place}"
            gdf = ox.geocode_to_gdf(place)
            return gdf, f"OSM boundary for {place}"
        if feature in ("road", "roads", "street", "highway"):
            graph = ox.graph_from_place(place, network_type="all")
            gdf = ox.graph_to_gdfs(graph, nodes=False)
            return gdf.reset_index(), f"OSM road network for {place}"
        gdf = ox.features_from_place(place, tags=tags)
        return gdf.reset_index(), f"OSM {feature} for {place}"

    def _download_natural_earth(self, place: str, query: str) -> Tuple[Any, str]:
        """Country/admin polygons from Natural Earth (public domain, global)."""
        import geopandas as gpd
        url = ("https://naturalearth.s3.amazonaws.com/110m_cultural/"
               "ne_110m_admin_0_countries.zip")
        gdf = gpd.read_file(url)
        if place:
            name_cols = [c for c in gdf.columns if c.upper() in
                         ("NAME", "ADMIN", "NAME_LONG", "SOVEREIGNT")]
            mask = None
            for c in name_cols:
                m = gdf[c].astype(str).str.contains(place, case=False, na=False)
                mask = m if mask is None else (mask | m)
            if mask is not None and mask.any():
                gdf = gdf[mask]
        return gdf, f"Natural Earth admin boundary for {place or 'world'}"

    # ---- entry point ------------------------------------------------------
    def run(self, query: str, input_paths: List[str],
            progress_callback: Optional[Callable]) -> dict:
        agent = self.agent
        params = self._parse_request(query)
        place = params["place"]
        agent._emit_progress(progress_callback, "discovery_parse",
                             f"Interpreting the download request "
                             f"(place={place!r}, feature={params['feature']!r}).",
                             {"params": params})
        if not place:
            raise ValueError("Could not determine a place/area to download. "
                             "Mention a place name, e.g. 'roads of Lisbon, Portugal'.")

        gdf, label = None, ""
        errors = []
        # Try OSM first (richest, global, open).
        try:
            agent._emit_progress(progress_callback, "discovery_osm",
                                 f"Querying OpenStreetMap for {place}.", {})
            gdf, label = self._download_osm(place, params["feature"], params["tags"],
                                            params["admin_only"], query)
        except Exception as exc:
            errors.append(f"OSM: {exc}")
            logger.warning("OSM download failed: %s", exc)

        # Fall back to Natural Earth for admin/country-level requests.
        if (gdf is None or len(gdf) == 0):
            try:
                agent._emit_progress(progress_callback, "discovery_ne",
                                     "Falling back to Natural Earth (admin boundaries).", {})
                gdf, label = self._download_natural_earth(place, query)
            except Exception as exc:
                errors.append(f"NaturalEarth: {exc}")

        if gdf is None or len(gdf) == 0:
            raise RuntimeError("No data could be downloaded. " + " | ".join(errors))

        # Keep only serialisable columns (OSM can include list-typed fields).
        for col in list(gdf.columns):
            if col != gdf.geometry.name:
                gdf[col] = gdf[col].apply(
                    lambda v: ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else v)

        ext = _preferred_vector_ext(query)
        out = self._save(gdf, query, ext)
        return {"text": f"Downloaded {len(gdf)} feature(s): {label}.",
                "dataset_paths": [out],
                "feature_count": len(gdf)}

    def _save(self, gdf, query: str, ext: str) -> str:
        path = self.agent._out_path(f"download {query}", ext, "download")
        return _write_vector(gdf, path)


def run(query: str, input_dataset_paths=None, *, agent=None, progress_callback=None,
        provider=None, model=None, api_key=None, output_dir=None) -> dict:
    """Run the data-discovery capability standalone (no orchestrator needed)."""
    host = agent or make_host(api_key=api_key, model=model,
                              output_dir=output_dir, provider=provider)
    cap = DataDiscoveryCapability(host)
    paths = host.normalize_dataset_paths(input_dataset_paths)
    return cap.run(query, paths, progress_callback)
