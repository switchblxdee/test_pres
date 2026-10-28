"""Пайплайн как граф LangGraph.

    START → load ──(Send × N тем)──► select_quotes ─┐
                 └──(нет кандидатов)────────────────┴─► assemble → render → END

select_quotes исполняется параллельно по темам (fan-out через Send), результаты
сливаются редьюсером в state["quotes"]. Параллелизм ограничен max_concurrency.
"""
from __future__ import annotations

import logging
import operator
import time
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .config import Config
from .data import Digest, Topic, load_digest
from .measure import Measurer
from .quotes import QuoteSelector
from .render import Layouter, Renderer

log = logging.getLogger(__name__)


def _merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}


class DigestState(TypedDict, total=False):
    input_path: str
    output_path: str
    digest: Digest
    quotes: Annotated[dict[int, list[str]], _merge_dicts]
    n_slides: int
    output: str
    timings: Annotated[list[tuple[str, float]], operator.add]


class TopicTask(TypedDict):
    idx: int
    topic: Topic


def build_graph(cfg: Config, llm: BaseChatModel | None = None):
    """llm можно подставить снаружи (тесты, свой клиент); иначе создаётся из cfg.llm."""
    selector = QuoteSelector(cfg, llm=llm)

    def load(state: DigestState) -> DigestState:
        t0 = time.perf_counter()
        digest = load_digest(state["input_path"], cfg)
        log.info("Свод: %d продукт(ов), %d тем, новых %d",
                 len(digest.products), digest.meta.active_topics, digest.meta.new_topics)
        return {"digest": digest, "quotes": {}, "timings": [("load", time.perf_counter() - t0)]}

    def fan_out(state: DigestState):
        tasks = [Send("select_quotes", TopicTask(idx=i, topic=t))
                 for i, t in enumerate(state["digest"].topics) if t.candidates]
        return tasks or "assemble"

    def select_quotes(task: TopicTask) -> DigestState:
        return {"quotes": {task["idx"]: selector.select(task["topic"])}}

    def assemble(state: DigestState) -> DigestState:
        digest = state["digest"]
        for i, t in enumerate(digest.topics):
            t.quotes = state["quotes"].get(i, [])
        selector.cache.save()
        return {"digest": digest}

    def render(state: DigestState) -> DigestState:
        t0 = time.perf_counter()
        measurer = Measurer(cfg.fonts)
        plans = Layouter(cfg, measurer).plan(state["digest"])
        out = Renderer(cfg, measurer).render(state["digest"], plans, state["output_path"])
        return {"output": str(out), "n_slides": len(plans), "timings": [("render", time.perf_counter() - t0)]}

    g = StateGraph(DigestState)
    g.add_node("load", load)
    g.add_node("select_quotes", select_quotes)
    g.add_node("assemble", assemble)
    g.add_node("render", render)
    g.add_edge(START, "load")
    g.add_conditional_edges("load", fan_out, ["select_quotes", "assemble"])
    g.add_edge("select_quotes", "assemble")
    g.add_edge("assemble", "render")
    g.add_edge("render", END)
    return g.compile()


def build(input_xlsx: str | Path, output_pptx: str | Path, cfg: Config,
          llm: BaseChatModel | None = None) -> Path:
    t0 = time.perf_counter()
    graph = build_graph(cfg, llm=llm)
    state: dict[str, Any] = graph.invoke(
        {"input_path": str(input_xlsx), "output_path": str(output_pptx)},
        config={"max_concurrency": max(1, cfg.llm.max_workers), "recursion_limit": 50},
    )
    log.info("Готово: %s, слайдов %d, %.1f с", state["output"], state["n_slides"], time.perf_counter() - t0)
    return Path(state["output"])
