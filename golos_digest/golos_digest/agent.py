"""DigestAgent — агент генерации дайджеста «Голос IT» на LangGraph.

    START → load ──(Send × N тем)──► select_quotes ─┐
                 └──(нет кандидатов)────────────────┴─► assemble → render → END

Использование:

    agent = DigestAgent.from_yaml("config.yaml")             # модель из конфига
    agent = DigestAgent(cfg, llm=GigaChat(...))              # или своя готовая модель
    result = agent.run("выгрузка.xlsx", "digest.pptx")
    print(result.path, result.n_slides)

Граф компилируется один раз в конструкторе; экземпляр можно переиспользовать
для нескольких запусков (каждый run — независимое состояние).
"""
from __future__ import annotations

import logging
import operator
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .config import Config, load_config
from .data import Digest, Topic, load_digest
from .measure import Measurer
from .quotes import QuoteSelector
from .sentences import Quote
from .render import Layouter, Renderer

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- состояние графа


def _merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}


class DigestState(TypedDict, total=False):
    input_path: str
    output_path: str
    digest: Digest
    quotes: Annotated[dict[int, list[Quote]], _merge_dicts]   # редьюсер для параллельных веток
    n_slides: int
    output: str
    timings: Annotated[list[tuple[str, float]], operator.add]


class TopicTask(TypedDict):
    idx: int
    topic: Topic


@dataclass(frozen=True)
class DigestResult:
    path: Path
    n_slides: int
    n_topics: int
    llm_calls: int          # тем, отправленных в LLM (не из кэша)
    llm_failures: int       # из них откатились на эвристику после всех ретраев
    seconds: float

    def __str__(self) -> str:
        s = f"{self.path}: слайдов {self.n_slides}, тем {self.n_topics}, {self.seconds:.1f} с"
        if self.llm_calls:
            s += f", LLM {self.llm_calls - self.llm_failures}/{self.llm_calls}"
        return s


# ----------------------------------------------------------------------------- агент


class DigestAgent:
    def __init__(self, config: Config | str | Path | None = None, llm: BaseChatModel | None = None):
        """config: объект Config, путь к YAML или None (значения по умолчанию).
        llm: готовая LangChain-модель; если не передана — создаётся из config.llm."""
        self.cfg = config if isinstance(config, Config) else load_config(config)
        self.selector = QuoteSelector(self.cfg, llm=llm)
        self.graph = self._build_graph()

    @classmethod
    def from_yaml(cls, path: str | Path, llm: BaseChatModel | None = None) -> "DigestAgent":
        return cls(load_config(path), llm=llm)

    @property
    def llm(self) -> BaseChatModel | None:
        return self.selector.llm

    # ---- узлы

    def _load(self, state: DigestState) -> DigestState:
        t0 = time.perf_counter()
        digest = load_digest(state["input_path"], self.cfg)
        log.info("Свод: %d продукт(ов), %d тем, новых %d",
                 len(digest.products), digest.meta.active_topics, digest.meta.new_topics)
        return {"digest": digest, "quotes": {}, "timings": [("load", time.perf_counter() - t0)]}

    @staticmethod
    def _fan_out(state: DigestState):
        tasks = [Send("select_quotes", TopicTask(idx=i, topic=t))
                 for i, t in enumerate(state["digest"].topics) if t.candidates]
        return tasks or "assemble"

    def _select_quotes(self, task: TopicTask) -> DigestState:
        return {"quotes": {task["idx"]: self.selector.select(task["topic"])}}

    def _assemble(self, state: DigestState) -> DigestState:
        digest = state["digest"]
        for i, t in enumerate(digest.topics):
            t.quotes = state["quotes"].get(i, [])
        self.selector.cache.save()
        return {"digest": digest}

    def _render(self, state: DigestState) -> DigestState:
        t0 = time.perf_counter()
        measurer = Measurer(self.cfg.fonts)
        plans = Layouter(self.cfg, measurer).plan(state["digest"])
        out = Renderer(self.cfg, measurer).render(state["digest"], plans, state["output_path"])
        return {"output": str(out), "n_slides": len(plans), "timings": [("render", time.perf_counter() - t0)]}

    def _build_graph(self):
        g = StateGraph(DigestState)
        g.add_node("load", self._load)
        g.add_node("select_quotes", self._select_quotes)
        g.add_node("assemble", self._assemble)
        g.add_node("render", self._render)
        g.add_edge(START, "load")
        g.add_conditional_edges("load", self._fan_out, ["select_quotes", "assemble"])
        g.add_edge("select_quotes", "assemble")
        g.add_edge("assemble", "render")
        g.add_edge("render", END)
        return g.compile()

    # ---- запуск

    def run(self, input_xlsx: str | Path, output_pptx: str | Path | None = None) -> DigestResult:
        """Синхронный запуск. Без output_pptx файл кладётся рядом с xlsx."""
        input_xlsx = Path(input_xlsx)
        if not input_xlsx.is_file():
            raise FileNotFoundError(f"Нет файла: {input_xlsx}")
        output_pptx = Path(output_pptx) if output_pptx else \
            input_xlsx.with_name(f"golos_it_digest_{date.today():%Y%m%d}.pptx")
        calls0, fails0 = self.selector.llm_calls, self.selector.llm_failures
        t0 = time.perf_counter()
        state = self.graph.invoke(
            {"input_path": str(input_xlsx), "output_path": str(output_pptx)},
            config={"max_concurrency": max(1, self.cfg.llm.max_workers), "recursion_limit": 50},
        )
        result = DigestResult(
            path=Path(state["output"]), n_slides=state["n_slides"], n_topics=len(state["digest"].topics),
            llm_calls=self.selector.llm_calls - calls0, llm_failures=self.selector.llm_failures - fails0,
            seconds=time.perf_counter() - t0,
        )
        log.info("Готово: %s", result)
        if result.llm_failures:
            log.warning("У %d тем LLM не ответил — цитаты выбраны эвристикой (в кэш не записаны, "
                        "следующий запуск попробует снова)", result.llm_failures)
        return result

    __call__ = run

    def draw(self) -> str:
        """Mermaid-схема графа (для README / отладки)."""
        return self.graph.get_graph().draw_mermaid()


def build(input_xlsx: str | Path, output_pptx: str | Path, cfg: Config,
          llm: BaseChatModel | None = None) -> Path:
    """Короткий вызов одной функцией (обёртка над DigestAgent)."""
    return DigestAgent(cfg, llm=llm).run(input_xlsx, output_pptx).path
