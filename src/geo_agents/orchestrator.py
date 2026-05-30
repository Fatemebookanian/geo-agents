"""
geo_agents.orchestrator
========================
THE UNIFIED ORCHESTRATOR -- the single agent you talk to.

``UnifiedGeoAgent`` registers every capability and routes a request to the right
one, chaining several into one workflow when needed. An LLM planner is used when
a model is reachable; otherwise it falls back to deterministic keyword routing,
so it always runs.

Notebook helpers:
    from geo_agents.orchestrator import get_agent, ask, launch_ui
    ask("download the parks of Tehran, Iran and make a web map")
    launch_ui()                       # interactive text box + Run button
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .common import BaseGeoAgent, _parse_llm_json
from .data_discovery import DataDiscoveryCapability
from .projection import ProjectionCapability
from .vector_analysis import VectorAnalysisCapability
from .raster import RasterCapability
from .spatial_statistics import SpatialStatisticsCapability
from .mapping import MappingCapability
from .web_mapping import WebMappingCapability

logger = logging.getLogger("geo_agents")

__all__ = ["UnifiedGeoAgent", "GeoSuiteAgent", "get_agent", "ask", "launch_ui"]


class UnifiedGeoAgent(BaseGeoAgent):
    agent_id = "unified_geo_agent"
    agent_name = "Unified Geo Agent"
    agent_version = "2.0.0"
    agent_description = (
        "A single self-contained agent that routes a request to the right GIS "
        "capability (global data discovery/download, reprojection, vector analysis, "
        "raster analysis, spatial statistics, static maps, interactive web maps) and "
        "can chain several of them into one workflow.")

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 output_dir: Optional[Path] = None, provider: Optional[str] = None,
                 base_url: Optional[str] = None):
        super().__init__(api_key=api_key, model=model, output_dir=output_dir,
                         provider=provider, base_url=base_url)
        # Register all capabilities in this single object.
        self.capabilities: Dict[str, Any] = {}
        for cap_cls in (DataDiscoveryCapability, ProjectionCapability,
                        VectorAnalysisCapability, RasterCapability,
                        SpatialStatisticsCapability, MappingCapability,
                        WebMappingCapability):
            cap = cap_cls(self)
            self.capabilities[cap.key] = cap
        self.lineage_steps: List[str] = []
        self.step_results: List[dict] = []

    # ----------------------------------------------------------- planning
    def _heuristic_route(self, query: str, has_datasets: bool) -> List[dict]:
        text = (query or "").lower()
        scores: Dict[str, int] = {}
        for key, cap in self.capabilities.items():
            score = sum(1 for kw in cap.keywords if kw in text)
            if score:
                scores[key] = score
        if any(t in text for t in ("download", "discover", "find data", "osm",
                                   "openstreetmap", "natural earth")):
            scores["data_discovery"] = scores.get("data_discovery", 0) + 3
        if any(t in text for t in ("interactive", "leaflet", "web map", "webmap")):
            scores["web_mapping"] = scores.get("web_mapping", 0) + 2
        if scores:
            best = max(scores, key=scores.get)
        elif not has_datasets:
            best = "data_discovery"
        else:
            best = "vector_analysis"
        return [{"capability": best, "instruction": query, "use_original_inputs": False}]

    def _llm_plan(self, query: str, has_datasets: bool) -> Optional[List[dict]]:
        if self.client is None:
            return None
        catalog = [{"capability": k, "description": c.description,
                    "produces_new_dataset": c.produces_data,
                    "requires_input_dataset": c.needs_input}
                   for k, c in self.capabilities.items()]
        raw = self._llm_chat(
            system="You are a senior GIS workflow planner. Use ONLY the provided "
                   "capabilities and keep the plan minimal.",
            user=json.dumps({
                "user_request": query,
                "user_already_has_input_datasets": has_datasets,
                "available_capabilities": catalog,
                "instructions": (
                    "Build the shortest ordered plan that fulfils the request. Each "
                    "step uses one capability key and an 'instruction' (short task "
                    "text). Steps run in order; by default each step receives the "
                    "dataset(s) produced by the previous step. Set 'use_original_inputs' "
                    "true for a step that should use the user's original input(s). If "
                    "the user has no data and needs some, start with 'data_discovery'. "
                    'Return ONLY JSON: {"plan":[{"capability":"...","instruction":"...",'
                    '"use_original_inputs":false}]}.'),
            }),
            purpose="planning",
        )
        payload = _parse_llm_json(raw) if raw else None
        if not payload or not isinstance(payload.get("plan"), list):
            return None
        cleaned = []
        for raw_step in payload["plan"]:
            if isinstance(raw_step, dict) and raw_step.get("capability") in self.capabilities:
                cleaned.append({
                    "capability": raw_step["capability"],
                    "instruction": raw_step.get("instruction") or query,
                    "use_original_inputs": bool(raw_step.get("use_original_inputs", False)),
                })
        return cleaned or None

    def _build_plan(self, query: str, has_datasets: bool) -> List[dict]:
        plan = self._llm_plan(query, has_datasets)
        if plan:
            self.lineage_steps.append(f"Planned {len(plan)} step(s) via LLM planner.")
            return plan
        plan = self._heuristic_route(query, has_datasets)
        self.lineage_steps.append("Planned via keyword heuristics.")
        return plan

    # --------------------------------------------------------------- run
    def run(self, query: str, input_dataset_paths: List[str] | str | None = None,
            progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
            max_iterations: int = 8) -> Dict[str, Any]:
        start = time.time()
        original_inputs = self.normalize_dataset_paths(input_dataset_paths)
        has_datasets = bool(original_inputs)
        self.input_tokens = self.output_tokens = self.llm_calls = self.tool_calls = 0
        self.lineage_steps, self.step_results = [], []

        agent_data = {
            "agent_name": self.agent_name, "agent_version": self.agent_version,
            "model": self.model, "duration": None,
            "total_input_tokens": 0, "total_output_tokens": 0,
            "inputs": {"text": query, "dataset_path": original_inputs},
            "outputs": {"text": None, "dataset_path": None, "dataset_paths": [],
                        "dataset_size": {"type": None, "feature_count": None}},
            "metrics": {"llm_calls": 0, "tool_calls": 0, "number_of_artifacts": 0},
            "environment": {"python_version": sys.version.split(" ")[0],
                            "capabilities": list(self.capabilities.keys())},
            "complementary": {"Execution": {"Inputs": {}, "Outputs": {}},
                              "Provenance": {"Plan": [], "Steps": [], "Lineage": {}}},
        }

        try:
            self._emit_progress(progress_callback, "start",
                                "Analysing the request and routing it to the right "
                                "GIS capability (chaining if needed).",
                                {"has_input_datasets": has_datasets})
            plan = self._build_plan(query, has_datasets)[:max_iterations]
            agent_data["complementary"]["Provenance"]["Plan"] = plan
            self._emit_progress(progress_callback, "plan_ready",
                                "Plan: " + " -> ".join(s["capability"] for s in plan),
                                {"plan": plan})

            current_inputs = list(original_inputs)
            all_artifacts: List[str] = []
            text_parts: List[str] = []
            last_size = None

            for idx, step in enumerate(plan, 1):
                cap = self.capabilities[step["capability"]]
                step_inputs = original_inputs if step.get("use_original_inputs") else current_inputs
                self._emit_progress(progress_callback, "step_start",
                                    f"Step {idx}/{len(plan)}: '{cap.key}'.",
                                    {"step": idx, "capability": cap.key,
                                     "instruction": step["instruction"]})
                self.tool_calls += 1
                try:
                    res = cap.run(step["instruction"], step_inputs, progress_callback)
                except Exception as step_exc:
                    logger.exception("Capability %s failed", cap.key)
                    res = {"text": f"[{cap.key}] failed: {step_exc}", "dataset_paths": []}

                arts = res.get("dataset_paths", []) or []
                if res.get("feature_count") is not None:
                    last_size = {"type": "vector",
                                 "feature_count": res.get("feature_count")}
                all_artifacts.extend(arts)
                if res.get("text"):
                    text_parts.append(f"[{cap.key}] {res['text']}")
                self.step_results.append({
                    "step": idx, "capability": cap.key,
                    "instruction": step["instruction"],
                    "output_text": res.get("text"), "artifacts": arts})
                self.lineage_steps.append(f"Step {idx}: {cap.key} -> {len(arts)} artifact(s).")
                # Chain: feed produced data into the next step.
                if arts:
                    data_arts = [a for a in arts if not a.lower().endswith((".png", ".html"))]
                    current_inputs = data_arts or current_inputs
                self._emit_progress(progress_callback, "step_complete",
                                    f"Step {idx} ('{cap.key}') done: {len(arts)} artifact(s).",
                                    {"artifacts": arts})

            seen = set()
            unified = [p for p in all_artifacts if not (p in seen or seen.add(p))]
            agent_data["outputs"]["dataset_paths"] = unified
            agent_data["outputs"]["dataset_path"] = unified[-1] if unified else None
            agent_data["outputs"]["text"] = "\n\n".join(text_parts) or "Workflow completed."
            if last_size:
                agent_data["outputs"]["dataset_size"] = last_size
            agent_data["metrics"]["number_of_artifacts"] = len(unified)
            self._emit_progress(progress_callback, "complete",
                                "Unified workflow complete.",
                                {"total_artifacts": len(unified)})

        except Exception as exc:
            logger.exception("UnifiedGeoAgent failed")
            self._emit_progress(progress_callback, "error", f"Workflow error: {exc}")
            agent_data["outputs"]["text"] = f"Status: failed. Error: {exc}"
            self.lineage_steps.append(f"ERROR: {exc}")

        agent_data["duration"] = f"{time.time() - start:.2f}s"
        agent_data["total_input_tokens"] = self.input_tokens
        agent_data["total_output_tokens"] = self.output_tokens
        agent_data["metrics"]["llm_calls"] = self.llm_calls
        agent_data["metrics"]["tool_calls"] = self.tool_calls
        agent_data["complementary"]["Execution"]["Inputs"] = {
            "text_query": query, "input_dataset_paths": original_inputs}
        agent_data["complementary"]["Execution"]["Outputs"] = {
            "text": agent_data["outputs"]["text"],
            "dataset_paths": agent_data["outputs"]["dataset_paths"]}
        agent_data["complementary"]["Provenance"]["Steps"] = self.step_results
        agent_data["complementary"]["Provenance"]["Lineage"] = {
            "steps": self.lineage_steps, "count": len(self.lineage_steps)}
        return agent_data


# Convenience alias.
GeoSuiteAgent = UnifiedGeoAgent


# =============================================================================
# Jupyter helpers
# =============================================================================
# A single shared agent so notebook cells can just call ask("...").
_DEFAULT_AGENT: Optional["UnifiedGeoAgent"] = None


def get_agent(**kwargs) -> "UnifiedGeoAgent":
    """Return a shared UnifiedGeoAgent, creating it on first use.

    Pass provider=/model=/api_key= once to configure it, e.g.
        get_agent(provider="groq")          # free cloud (needs GROQ_API_KEY)
        get_agent(provider="ollama")        # free + local, no key (default)
    """
    global _DEFAULT_AGENT
    if _DEFAULT_AGENT is None or kwargs:
        _DEFAULT_AGENT = UnifiedGeoAgent(**kwargs)
    return _DEFAULT_AGENT


def ask(request: str, input_dataset_paths=None, verbose: bool = True) -> dict:
    """One-liner for notebooks: just write your request as text.

        ask("buffer roads.gpkg by 100 m")
        ask("download the parks of Tehran, Iran and make a web map")

    Returns the full result dict; also prints a short summary when verbose.
    """
    agent = get_agent()

    def _cb(ev):
        if verbose and ev.get("message"):
            print(f"  … {ev['message']}")

    result = agent.run(request, input_dataset_paths=input_dataset_paths,
                       progress_callback=_cb if verbose else None)
    if verbose:
        print("\n" + (result["outputs"]["text"] or ""))
        if result["outputs"]["dataset_paths"]:
            print("\nArtifacts:")
            for p in result["outputs"]["dataset_paths"]:
                print("  -", p)
    return result


def launch_ui(**agent_kwargs):
    """Render an interactive panel (text box + Run button) inside Jupyter.

    Type your request in the box and click Run -- results show below.
    Requires ipywidgets:  pip install ipywidgets
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output, HTML
    except Exception as exc:
        print("launch_ui needs ipywidgets and a Jupyter environment. "
              f"Install with: pip install ipywidgets   ({exc})")
        return

    agent = get_agent(**agent_kwargs) if agent_kwargs else get_agent()

    box = widgets.Textarea(
        placeholder="e.g. download the road network of Nuremberg, Germany, "
                    "reproject to UTM, buffer 50 m, and build a web map",
        layout=widgets.Layout(width="100%", height="80px"))
    files = widgets.Text(
        placeholder="optional: input file path(s), comma-separated",
        layout=widgets.Layout(width="100%"))
    run_btn = widgets.Button(description="Run", button_style="primary", icon="play")
    out = widgets.Output()

    def _on_click(_):
        with out:
            clear_output()
            req = box.value.strip()
            if not req:
                print("Type a request first.")
                return
            paths = [p.strip() for p in files.value.split(",") if p.strip()] or None
            print(f"Provider: {agent.provider} | model: {agent.model}\n")
            res = agent.run(req, input_dataset_paths=paths,
                            progress_callback=lambda e: print("  …", e.get("message", "")))
            print("\n=== RESULT ===")
            print(res["outputs"]["text"])
            for p in res["outputs"]["dataset_paths"]:
                print("artifact:", p)

    run_btn.on_click(_on_click)
    display(widgets.VBox([widgets.HTML("<b>Unified Geo Agent</b>"),
                          box, files, run_btn, out]))


if __name__ == "__main__":
    agent = UnifiedGeoAgent()
    print("Capabilities:", list(agent.capabilities.keys()))
    print("LLM provider:", agent.provider, "| model:", agent.model)
    demo = ("download the road network of Nuremberg Germany, reproject to UTM, "
            "buffer 50 m, then build an interactive web map")
    print("Heuristic route:",
          [s["capability"] for s in agent._heuristic_route(demo, has_datasets=False)])
