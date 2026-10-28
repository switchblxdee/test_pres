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


@pytest.mark.parametrize("raw,expected", [
    ("<think>Сравню {1} и [3]: 3 хуже, 1 лучше</think>\n{\"ranking\": [2, 1]}", [2, 1]),
    ("<think>долго думаю про 7 и 9", None),  # обрыв внутри рассуждений — разбирать нечего
    ("Итог: [3, 1]", [3, 1]),
])
def test_parse_ranking_reasoning_models(raw, expected):
    if expected is None:
        with pytest.raises(ValueError):
            parse_ranking(raw)
    else:
        assert parse_ranking(raw).ranking == expected


def test_structured_output_off_single_call(cfg):
    cfg.llm.backend, cfg.llm.retries, cfg.llm.structured_output = "gigachat", 1, False
    sel = QuoteSelector(cfg, llm=FakeModel(mode="text"))
    t = topic()
    assert sel.select(t)[0] == prefilter(t, cfg.quotes, 40)[1]
    assert sel.llm.calls == 1  # без попытки function calling


def test_gigachat_factory_mtls(tmp_path, monkeypatch):
    from golos_digest.config import LLMConfig
    from golos_digest.quotes import make_chat_model
    monkeypatch.delenv("GIGACHAT_CREDENTIALS", raising=False)
    files = {}
    for name in ("serv.pem", "client.pem", "client.key"):
        files[name] = tmp_path / name
        files[name].write_text("x")
    c = LLMConfig(ca_bundle_file=str(files["serv.pem"]), cert_file=str(files["client.pem"]),
                  key_file=str(files["client.key"]), extra={"max_tokens": 256})
    m = make_chat_model(c)
    assert (m.model, m.base_url, m.temperature, m.repetition_penalty, m.verify_ssl_certs) == \
        ("Qwen3.5-397b", "https://gigachat-ift.sberdevices.delta.sbrf.ru/v1", 0.0, 1.05, True)
    assert m.cert_file == str(files["client.pem"]) and m.credentials is None and m.max_tokens == 256


def test_gigachat_factory_errors(tmp_path, monkeypatch):
    from golos_digest.config import LLMConfig
    from golos_digest.quotes import make_chat_model
    monkeypatch.delenv("GIGACHAT_CREDENTIALS", raising=False)
    with pytest.raises(RuntimeError, match="Нет авторизации"):
        make_chat_model(LLMConfig())
    with pytest.raises(RuntimeError, match="Не найден cert_file"):
        make_chat_model(LLMConfig(cert_file=str(tmp_path / "nope.pem"), key_file=str(tmp_path / "k")))
