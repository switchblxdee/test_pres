import threading

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from golos_digest.data import Topic
from golos_digest.quotes import QuoteSelector, normalize_ranking, parse_ranking, prefilter

_lock = threading.Lock()


class FakeModel(BaseChatModel):
    mode: str = "tool"  # tool | text | fail
    calls: int = 0

    @property
    def _llm_type(self):
        return "fake"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        with _lock:
            self.calls += 1
        if self.mode == "fail":
            raise ConnectionError("503")
        if self.mode == "text":
            msg = AIMessage(content='Ответ:\n```json\n{"ranking": [2, 1]}\n```')
        else:
            msg = AIMessage(content="", tool_calls=[{"name": "Ranking", "args": {"ranking": [2, 1]}, "id": "1"}])
        return ChatResult(generations=[ChatGeneration(message=msg)])


def topic():
    texts = [
        "Сервис падает с ошибкой 500 при загрузке файла, повторные попытки не помогают совсем.",
        "Не могу подключиться к MCP: таймаут и ошибка 401, раньше всё работало нормально.",
        "+1", "у меня тоже",
    ]
    return Topic("P", "Ошибки подключения MCP", 5, 4, -1, "", None, False, False, candidates=texts)


@pytest.mark.parametrize("raw,expected", [
    ('{"ranking": [3, 1]}', [3, 1]),
    ('```json\n{"ranking": [2]}\n```', [2]),
    ("Лучшие: 4 и 2", [4, 2]),
])
def test_parse_ranking(raw, expected):
    assert parse_ranking(raw).ranking == expected


def test_normalize_ranking_dedup_and_bounds():
    assert normalize_ranking([2, 2, 9, 1], 3) == [1, 0, 2]


def test_prefilter_drops_noise():
    c = prefilter(topic(), __import__("golos_digest.config").config.QuoteConfig(), 10)
    assert "+1" not in c and "у меня тоже" not in c and len(c) == 2


@pytest.mark.parametrize("mode,calls,picked_second,cached", [
    ("tool", 1, True, True),     # structured output
    ("text", 2, True, True),     # structured вернул None -> текстовый фолбэк
    ("fail", 2, False, False),   # ретраи -> эвристика, сбой не кэшируется
])
def test_selector_modes(cfg, mode, calls, picked_second, cached):
    cfg.llm.backend, cfg.llm.retries = "gigachat", 1
    t = topic()
    sel = QuoteSelector(cfg, llm=FakeModel(mode=mode))
    cands = prefilter(t, cfg.quotes, cfg.llm.max_candidates)
    picked = sel.select(t)
    assert sel.llm.calls == calls
    assert (picked[0] == cands[1]) is picked_second
    assert bool(sel.cache.data) is cached
