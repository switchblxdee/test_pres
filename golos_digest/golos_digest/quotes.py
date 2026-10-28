"""Отбор цитат: эвристический префильтр + ранжирование LLM через LangChain.

LLM только выбирает номер сообщения — текст цитаты всегда дословный из разметки,
поэтому галлюцинации в цитатах исключены by design.

Цепочка ранжирования (LCEL):
    prompt | llm.with_structured_output(Ranking)        # function calling GigaChat
      .with_fallbacks([prompt | llm | text -> Ranking])  # если модель ответила текстом
      .with_retry(...)
При полном отказе LLM узел графа откатывается на эвристику.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

from .config import Config, LLMConfig, QuoteConfig
from .data import Topic, norm

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Ты — аналитик голоса клиента. Твоя задача — выбрать из сообщений пользователей "
    "цитату для еженедельного дайджеста руководству. Отвечай строго JSON без пояснений вокруг."
)

USER_PROMPT = """Продукт: {product}
Тема (кластер обращений): {topic}

Ниже пронумерованные сообщения пользователей по этой теме:

{messages}

Выбери сообщения, которые лучше всего подходят как цитата для слайда. Критерии по убыванию важности:
1. Точно про эту тему, а не про соседнюю проблему.
2. Полнота: понятно, что именно не работает или чего не хватает, где и к чему это приводит.
3. Самодостаточность: понятно без контекста переписки; не «+1», не «у меня тоже», не голое «подскажите».
4. Конкретика: версии, названия инструментов, коды ошибок, реальные последствия для работы.
5. Длина желательно от {min_chars} до {max_chars} символов; длиннее — только если сообщение заметно сильнее остальных.
6. Не выбирай сообщения, состоящие в основном из логов, ссылок, кода или персональных данных.

Верни JSON строго такого вида: {{"ranking": [номера лучших сообщений по убыванию качества, не больше 3]}}"""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", USER_PROMPT)])


class Ranking(BaseModel):
    """Номера лучших сообщений для цитаты, от лучшего к худшему."""

    ranking: list[int] = Field(description="Номера сообщений (1-based) по убыванию качества, максимум 3")


# ----------------------------------------------------------------------------- эвристика

_PROBLEM = re.compile(
    r"не\s+работа|не\s+могу|не\s+удается|не\s+удаётся|ошибк|сбо[йи]|падает|упал|"
    r"нет\s+доступ|доступ|долго|медленн|завис|таймаут|timeout|\b40[0-9]\b|\b50[0-9]\b|"
    r"проблем|невозможно|отказ|не\s+подключ|не\s+открыва|когда\s+(?:будет|смогут|появ)",
    re.I,
)
_NOISE = re.compile(r"^\W*(\+1|у\s+меня\s+(тоже|аналогично)|аналогично|спасибо|подскажите\W*)\W*$", re.I)
_URL = re.compile(r"https?://\S+")


def _stems(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-zа-яё0-9-]{4,}", text.lower())}


def heuristic_score(text: str, topic: str, q: QuoteConfig) -> float:
    n = len(text)
    if n < q.min_chars or _NOISE.match(text):
        return -1e9
    score = 0.0
    if q.ideal_min_chars <= n <= q.ideal_max_chars:
        score += 3.0
    elif n < q.ideal_min_chars:
        score += 3.0 * n / q.ideal_min_chars - 1.0
    else:
        score += 3.0 - (n - q.ideal_max_chars) / q.ideal_max_chars * 4.0
    score += min(len(_PROBLEM.findall(text)), 3) * 1.0
    score += min(len(_stems(text) & _stems(topic)), 4) * 0.8
    if "?" in text:
        score += 0.3
    score -= 0.7 * len(_URL.findall(text))
    letters = sum(c.isalpha() for c in text)
    if letters / max(n, 1) < 0.55:  # логи / код / стектрейсы
        score -= 3.0
    return score


def dedupe(texts: list[str]) -> list[str]:
    seen, out = set(), []
    for t in texts:
        k = re.sub(r"\W+", "", norm(t))
        if k and k not in seen:
            seen.add(k)
            out.append(t)
    return out


def prefilter(topic: Topic, q: QuoteConfig, limit: int) -> list[str]:
    pool = dedupe(topic.candidates)
    scored = [(heuristic_score(t, topic.title, q), t) for t in pool]
    scored = [st for st in scored if st[0] > -1e8] or [(0.0, t) for t in pool]
    scored.sort(key=lambda st: -st[0])
    return [t for _, t in scored[:limit]]


# ----------------------------------------------------------------------------- модели LangChain


def _check_file(path: str | None, what: str) -> str | None:
    if path and not Path(path).expanduser().is_file():
        raise RuntimeError(f"Не найден {what}: {path}")
    return str(Path(path).expanduser()) if path else None


def make_chat_model(c: LLMConfig) -> BaseChatModel | None:
    """Модель из конфига. Для своего, уже созданного клиента — DigestAgent(cfg, llm=model)."""
    if c.backend == "heuristic":
        return None
    if c.backend == "gigachat":
        try:
            from langchain_gigachat.chat_models import GigaChat
        except ImportError:
            raise RuntimeError("Не установлен langchain-gigachat: pip install langchain-gigachat "
                               "(или запусти с --backend heuristic)") from None
        kwargs: dict = dict(
            model=c.model, base_url=c.base_url, temperature=c.temperature, timeout=c.timeout,
            verify_ssl_certs=c.verify_ssl_certs, max_retries=0,  # ретраи — на уровне цепочки
            ca_bundle_file=_check_file(c.ca_bundle_file, "ca_bundle_file"),
            cert_file=_check_file(c.cert_file, "cert_file"),
            key_file=_check_file(c.key_file, "key_file"),
            repetition_penalty=c.repetition_penalty,
        )
        if os.environ.get(c.key_file_password_env):
            kwargs["key_file_password"] = os.environ[c.key_file_password_env]
        creds = os.environ.get(c.credentials_env)
        if creds:
            kwargs["credentials"] = creds
            if c.scope:
                kwargs["scope"] = c.scope
        elif not (c.cert_file and c.key_file):
            raise RuntimeError(f"Нет авторизации GigaChat: укажи llm.cert_file + llm.key_file (mTLS) "
                               f"или задай переменную окружения {c.credentials_env}")
        kwargs.update(c.extra)
        return GigaChat(**{k: v for k, v in kwargs.items() if v is not None})
    if c.backend == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise RuntimeError("Не установлен langchain-openai: pip install langchain-openai") from None
        return ChatOpenAI(model=c.model, base_url=c.base_url, temperature=c.temperature, timeout=c.timeout,
                          api_key=os.environ.get(c.api_key_env, "EMPTY"), max_retries=0, **c.extra)
    raise ValueError(f"Неизвестный llm.backend: {c.backend}")


_THINK = re.compile(r"<think>.*?(</think>|$)", flags=re.S | re.I)


def parse_ranking(raw: str) -> Ranking:
    """Текстовый ответ -> Ranking. Терпим к ```json```, тексту вокруг и блоку рассуждений
    <think>…</think> (Qwen3 и др.), в котором могут быть свои цифры и скобки."""
    text = _THINK.sub("", raw or "")
    ids: list[int] = []
    for blob in reversed(re.findall(r"\{[^{}]*\}", text)):  # последний JSON-объект — итоговый
        try:
            data = json.loads(blob)
            ids = [int(x) for x in data.get("ranking", data.get("best_ids", []))]
        except (ValueError, TypeError, AttributeError):
            continue
        if ids:
            break
    if not ids:
        m = re.search(r"\[([\d\s,]+)\]", text)  # голый массив [2, 1]
        ids = [int(x) for x in re.findall(r"\d+", m.group(1))] if m else []
    if not ids:
        ids = [int(x) for x in re.findall(r"\d+", text)[:3]]
    if not ids:
        raise ValueError(f"Не удалось разобрать ответ модели: {(raw or '')[:200]!r}")
    return Ranking(ranking=ids)


def _require(res: Ranking | None) -> Ranking:
    # with_structured_output возвращает None, если модель ответила текстом без tool call —
    # превращаем это в исключение, чтобы сработал with_fallbacks на текстовый парсер
    if res is None or not res.ranking:
        raise ValueError("пустой structured-ответ")
    return res


def build_ranker(llm: BaseChatModel, retries: int, structured: bool = True) -> Runnable:
    text_chain = PROMPT | llm | StrOutputParser() | RunnableLambda(parse_ranking)
    chain: Runnable = text_chain
    if structured:
        try:
            chain = (PROMPT | llm.with_structured_output(Ranking) | RunnableLambda(_require)
                     ).with_fallbacks([text_chain])
        except NotImplementedError:  # модель без tool calling
            chain = text_chain
    return chain.with_retry(stop_after_attempt=max(1, retries), wait_exponential_jitter=True)


def normalize_ranking(ids: list[int], n: int) -> list[int]:
    """1-based -> 0-based, без дублей и выходов за диапазон, дополняя хвостом эвристики."""
    out: list[int] = []
    for i in ids:
        if 1 <= i <= n and i - 1 not in out:
            out.append(i - 1)
    return out + [i for i in range(n) if i not in out]


# ----------------------------------------------------------------------------- кэш


class QuoteCache:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        try:
            self.data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}

    def get(self, key: str):
        return self.data.get(key)

    def put(self, key: str, value) -> None:
        with self._lock:
            self.data[key] = value

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)


# ----------------------------------------------------------------------------- селектор (одна тема)


class QuoteSelector:
    """Отбор цитат для одной темы; вызывается из параллельных веток графа, потокобезопасен."""

    def __init__(self, cfg: Config, llm: BaseChatModel | None = None):
        self.cfg = cfg
        self.llm = llm if llm is not None else make_chat_model(cfg.llm)
        self.ranker = (build_ranker(self.llm, cfg.llm.retries, cfg.llm.structured_output)
                       if self.llm is not None else None)
        self.cache = QuoteCache(Path(cfg.llm.cache_dir) / "quotes.json")
        self.llm_calls = 0      # темы, ушедшие в LLM (без кэша)
        self.llm_failures = 0   # из них откатились на эвристику
        self._stats_lock = threading.Lock()
        name = getattr(self.llm, "model", None) or getattr(self.llm, "model_name", None) or cfg.llm.model
        self.model_tag = f"{type(self.llm).__name__}:{name}" if self.llm is not None else "heuristic"

    def _key(self, topic: Topic, cands: list[str]) -> str:
        h = hashlib.sha1()
        for part in (self.model_tag, topic.product, topic.title, SYSTEM_PROMPT, USER_PROMPT, *cands):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def _rank(self, topic: Topic, cands: list[str]) -> tuple[list[int], bool]:
        """(ranking, надёжный ли результат). Фолбэк после сбоя LLM не кэшируется."""
        if self.ranker is None or len(cands) == 1:
            return list(range(len(cands))), True
        lim = self.cfg.llm.candidate_max_chars
        messages = "\n\n".join(
            f"[{i}] " + (c if len(c) <= lim else c[:lim] + "…") for i, c in enumerate(cands, 1)
        )
        with self._stats_lock:
            self.llm_calls += 1
        try:
            res: Ranking = self.ranker.invoke({
                "product": topic.product, "topic": topic.title, "messages": messages,
                "min_chars": self.cfg.quotes.ideal_min_chars, "max_chars": self.cfg.quotes.ideal_max_chars,
            })
            return normalize_ranking(res.ranking, len(cands)), True
        except Exception as e:  # сеть, 429, 5xx, токен, мусор в ответе — после всех ретраев
            with self._stats_lock:
                self.llm_failures += 1
            log.warning("LLM не ответил для темы «%s» (%s) — беру эвристику", topic.title, e)
            return list(range(len(cands))), False

    def select(self, topic: Topic) -> list[str]:
        """Топ-3 цитаты в порядке ранга; сколько из них встанет на слайд, решает вёрстка."""
        cands = prefilter(topic, self.cfg.quotes, self.cfg.llm.max_candidates)
        if not cands:
            return []
        key = self._key(topic, cands)
        ranking = self.cache.get(key)
        if ranking is None or any(i >= len(cands) for i in ranking):
            ranking, ok = self._rank(topic, cands)
            if ok:
                self.cache.put(key, ranking)
        return [cands[i] for i in ranking[:3]]
