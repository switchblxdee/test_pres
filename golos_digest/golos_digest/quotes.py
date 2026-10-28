"""Отбор цитат: эвристика + LLM через LangChain, на уровне ПРЕДЛОЖЕНИЙ.

Сообщения режутся на предложения (sentences.py) и уходят в LLM пронумерованными:
модель выбирает сообщение и номера предложений, которые лучше всего описывают проблему
пользователя (без приветствий, окружения, благодарностей). Текст цитаты собирается из
наших же предложений — дословно, галлюцинации исключены by design.

Цепочка (LCEL):
    prompt | llm.with_structured_output(Selection)        # function calling
      .with_fallbacks([prompt | llm | text -> Selection])  # ответ текстом / с <think>
      .with_retry(...)
При полном отказе LLM — эвристика: лучший фрагмент по оценке «проблемности» предложений.
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
from .sentences import SPLITTER_VERSION, Quote, best_quote, quote_from_ids, split_sentences

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Ты — аналитик голоса клиента. Из обращений пользователей ты выбираешь цитаты для еженедельного "
    "дайджеста руководству: фрагмент, по которому сразу понятна проблема пользователя. Цитата берётся "
    "дословно — ты только указываешь номера сообщения и предложений. "
    "Отвечай строго JSON без пояснений вокруг."
)

USER_PROMPT = """Продукт: {product}
Тема (кластер обращений): {topic}

Сообщения пользователей по этой теме. Каждое разбито на пронумерованные предложения, в скобках — длина в символах:

{messages}

Выбери до {n_quotes} цитат(ы) из РАЗНЫХ сообщений, лучшую первой. Цитата — одно или несколько предложений
одного сообщения, которые лучше всего описывают ПРОБЛЕМУ пользователя:
- что именно не работает или чего не хватает;
- где и при каких условиях (инструмент, действие, код ошибки — если это нужно для понимания);
- к чему это приводит в работе (если сказано).

НЕ включай в цитату:
- приветствия, благодарности, «+1», «у меня тоже», просьбы о помощи без сути;
- описание рабочего места, окружения, версий, должности, предыстории — если проблема понятна и без них;
- логи, код, ссылки, персональные данные.

Бери предложения по возможности подряд, суммарно до {max_chars} символов. Если проблема полностью
описана одним предложением — бери одно. Номера предложений перечисляй по важности: первым то, без
которого проблема непонятна.
Сообщения сравнивай по тому, насколько точно, полно и конкретно они описывают проблему именно этой темы.

Ответ строго JSON: {{"quotes": [{{"message": <номер сообщения>, "sentences": [<номера предложений по важности>]}}]}}"""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", USER_PROMPT)])


class Extract(BaseModel):
    """Цитата: номер сообщения и номера его предложений (по важности)."""

    message: int = Field(description="Номер сообщения (1-based)")
    sentences: list[int] = Field(default_factory=list,
                                 description="Номера предложений этого сообщения (1-based), самое важное первым")


class Selection(BaseModel):
    """Лучшие цитаты из разных сообщений, лучшая первой."""

    quotes: list[Extract] = Field(description="Цитаты, лучшая первой")


# ----------------------------------------------------------------------------- эвристика

_NOISE = re.compile(r"^\W*(\+\s*1|у\s+меня\s+(тоже|аналогично)|аналогично|спасибо|подскажите\W*)\W*$", re.I)


def dedupe(texts: list[str]) -> list[str]:
    seen, out = set(), []
    for t in texts:
        k = re.sub(r"\W+", "", norm(t))
        if k and k not in seen:
            seen.add(k)
            out.append(t)
    return out


def message_score(text: str, topic: str, q: QuoteConfig) -> float:
    """Качество сообщения = ценность его лучшего фрагмента (длина сообщения не штрафуется:
    из длинного письма возьмём только нужные предложения)."""
    if len(text) < q.min_chars or _NOISE.match(text):
        return float("-inf")
    return best_quote(text, topic, q.ideal_max_chars)[1]


def prefilter(topic: Topic, q: QuoteConfig, limit: int) -> list[str]:
    pool = dedupe(topic.candidates)
    scored = [(message_score(t, topic.title, q), t) for t in pool]
    useful = [st for st in scored if st[0] > float("-inf")] or [(0.0, t) for t in pool]
    useful.sort(key=lambda st: -st[0])
    return [t for _, t in useful[:limit]]


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
_DECODER = json.JSONDecoder()


def _json_objects(text: str) -> list:
    """Все JSON-объекты верхнего уровня в тексте (вложенность поддерживается)."""
    out, i = [], 0
    while (i := text.find("{", i)) != -1:
        try:
            obj, end = _DECODER.raw_decode(text, i)
            out.append(obj)
            i = end
        except ValueError:
            i += 1
    return out


def parse_selection(raw: str) -> Selection:
    """Текстовый ответ -> Selection. Терпим к ```json```, тексту вокруг, блоку <think>…</think>,
    одиночному объекту {"message": …} и старому формату {"ranking": […]}."""
    text = _THINK.sub("", raw or "")
    for obj in reversed(_json_objects(text)):  # последний объект — итоговый
        if not isinstance(obj, dict):
            continue
        try:
            if isinstance(obj.get("quotes"), list) and obj["quotes"]:
                return Selection(quotes=[Extract(**q) if isinstance(q, dict) else Extract(message=int(q))
                                         for q in obj["quotes"]])
            if "message" in obj:
                return Selection(quotes=[Extract(**obj)])
            if isinstance(obj.get("ranking"), list) and obj["ranking"]:
                return Selection(quotes=[Extract(message=int(m)) for m in obj["ranking"]])
        except (TypeError, ValueError):
            continue
    pairs = re.findall(r"\b(\d+)\s*[.:]\s*(\d+)\b", text)  # «3.2, 3.4»
    if pairs:
        m0 = int(pairs[0][0])
        return Selection(quotes=[Extract(message=m0, sentences=[int(s) for m, s in pairs if int(m) == m0])])
    nums = re.findall(r"\d+", text)
    if nums:
        return Selection(quotes=[Extract(message=int(nums[0]))])
    raise ValueError(f"Не удалось разобрать ответ модели: {(raw or '')[:200]!r}")


def _require(res: Selection | None) -> Selection:
    # with_structured_output возвращает None, если модель ответила текстом без tool call —
    # превращаем это в исключение, чтобы сработал with_fallbacks на текстовый парсер
    if res is None or not res.quotes:
        raise ValueError("пустой structured-ответ")
    return res


def build_extractor(llm: BaseChatModel, retries: int, structured: bool = True) -> Runnable:
    text_chain = PROMPT | llm | StrOutputParser() | RunnableLambda(parse_selection)
    chain: Runnable = text_chain
    if structured:
        try:
            chain = (PROMPT | llm.with_structured_output(Selection) | RunnableLambda(_require)
                     ).with_fallbacks([text_chain])
        except NotImplementedError:  # модель без tool calling
            chain = text_chain
    return chain.with_retry(stop_after_attempt=max(1, retries), wait_exponential_jitter=True)


def normalize_selection(sel: Selection, shown: list[list[str]]) -> list[list]:
    """1-based -> 0-based; отбрасывает несуществующие номера и повторы сообщений.
    Результат: [[message, [sentence, ...]], ...] — так же хранится в кэше."""
    out: list[list] = []
    used: set[int] = set()
    for e in sel.quotes:
        m = e.message - 1
        if not 0 <= m < len(shown) or m in used:
            continue
        used.add(m)
        sids = list(dict.fromkeys(s - 1 for s in e.sentences if 1 <= s <= len(shown[m])))
        out.append([m, sids])
    return out


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
        self.extractor = (build_extractor(self.llm, cfg.llm.retries, cfg.llm.structured_output)
                          if self.llm is not None else None)
        self.cache = QuoteCache(Path(cfg.llm.cache_dir) / "quotes.json")
        self.llm_calls = 0      # темы, ушедшие в LLM (без кэша)
        self.llm_failures = 0   # из них откатились на эвристику
        self._stats_lock = threading.Lock()
        name = getattr(self.llm, "model", None) or getattr(self.llm, "model_name", None) or cfg.llm.model
        self.model_tag = f"{type(self.llm).__name__}:{name}" if self.llm is not None else "heuristic"

    @property
    def n_quotes(self) -> int:
        return max(1, self.cfg.quotes.max_quotes)

    @property
    def budget(self) -> int:
        return self.cfg.quotes.ideal_max_chars

    def _shown(self, cands: list[str]) -> list[list[str]]:
        """Предложения сообщений в том виде, в каком их видит LLM (длинные письма — с начала,
        целыми предложениями до candidate_max_chars)."""
        lim, out = self.cfg.llm.candidate_max_chars, []
        for c in cands:
            sents, total = [], 0
            for s in split_sentences(c) or [c.strip()]:
                if sents and total + len(s) > lim:
                    break
                sents.append(s)
                total += len(s) + 1
            out.append(sents)
        return out

    @staticmethod
    def _format(shown: list[list[str]]) -> str:
        blocks = []
        for i, sents in enumerate(shown, 1):
            lines = "\n".join(f"  {j}. ({len(s)}) {s}" for j, s in enumerate(sents, 1))
            blocks.append(f"[{i}]\n{lines}")
        return "\n\n".join(blocks)

    def _key(self, topic: Topic, cands: list[str]) -> str:
        h = hashlib.sha1()
        for part in (self.model_tag, topic.product, topic.title, SYSTEM_PROMPT, USER_PROMPT, SPLITTER_VERSION,
                     str(self.budget), str(self.n_quotes), str(self.cfg.llm.candidate_max_chars), *cands):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def _extract(self, topic: Topic, shown: list[list[str]]) -> tuple[list[list], bool]:
        """(выбор LLM, надёжный ли результат). Сбой LLM не кэшируется."""
        with self._stats_lock:
            self.llm_calls += 1
        try:
            sel: Selection = self.extractor.invoke({
                "product": topic.product, "topic": topic.title, "messages": self._format(shown),
                "n_quotes": self.n_quotes, "max_chars": self.budget,
            })
            picks = normalize_selection(sel, shown)
            if picks:
                return picks, True
            raise ValueError("в ответе нет ни одного существующего сообщения")
        except Exception as e:  # сеть, 429, 5xx, мусор в ответе — после всех ретраев
            with self._stats_lock:
                self.llm_failures += 1
            log.warning("LLM не ответил для темы «%s» (%s) — беру эвристику", topic.title, e)
            return [], False

    def select(self, topic: Topic) -> list[Quote]:
        """До max_quotes цитат из разных сообщений, лучшая первой. Каждая — целые предложения."""
        cands = prefilter(topic, self.cfg.quotes, self.cfg.llm.max_candidates)
        if not cands:
            return []
        shown = self._shown(cands)
        picks: list[list] = []
        if self.extractor is not None and (len(cands) > 1 or len(shown[0]) > 1):
            key = self._key(topic, cands)
            cached = self.cache.get(key)
            valid = isinstance(cached, list) and all(
                isinstance(p, list) and len(p) == 2 and 0 <= p[0] < len(shown)
                and all(0 <= s < len(shown[p[0]]) for s in p[1]) for p in cached)
            if valid and cached:
                picks = cached
            else:
                picks, ok = self._extract(topic, shown)
                if ok:
                    self.cache.put(key, picks)

        quotes: list[Quote] = []
        used: set[int] = set()
        for m, sids in picks:
            q = quote_from_ids(shown[m], sids, topic.title, self.budget) if sids else None
            quotes.append(q or best_quote(cands[m], topic.title, self.budget)[0])
            used.add(m)
            if len(quotes) >= self.n_quotes:
                break
        for m, c in enumerate(cands):  # добор эвристикой (нет LLM / сбой / мало цитат)
            if len(quotes) >= self.n_quotes:
                break
            if m not in used:
                quotes.append(best_quote(c, topic.title, self.budget)[0])
        return quotes
