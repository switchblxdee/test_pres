import json
import threading

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from golos_digest.data import Topic
from golos_digest.quotes import (
    QuoteSelector, Selection, normalize_selection, parse_selection, prefilter,
)

_lock = threading.Lock()

ENV_FIRST = (
    "Добрый день! У меня рабочий стол на VDI, Windows 10, стоит VS Code 1.85 и GigaCode CLI версии 26.9.64. "
    "Я системный аналитик в ДЗО. При подключении MCP confluence-delta после 2-3 запросов получаю ошибку 401, "
    "токен при этом валидный. Из-за этого приходится каждый час перевыпускать токен вручную. Спасибо!"
)
SHORT = "Не могу подключиться к MCP: таймаут и ошибка 401, раньше всё работало нормально."


class FakeModel(BaseChatModel):
    """mode: tool — function call; text — JSON текстом с <think>; fail — исключение."""
    mode: str = "tool"
    calls: int = 0
    answer: dict = {"quotes": [{"message": 1, "sentences": [4, 5]}]}

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
            msg = AIMessage(content=f"<think>сравню {{1}} и [2]</think>\n```json\n{json.dumps(self.answer)}\n```")
        else:
            msg = AIMessage(content="", tool_calls=[{"name": "Selection", "args": self.answer, "id": "1"}])
        return ChatResult(generations=[ChatGeneration(message=msg)])


def topic():
    return Topic("GigaCode", "Ошибки авторизации MCP confluence", 5, 4, -1, "", None, False, False,
                 candidates=[ENV_FIRST, SHORT, "+1", "у меня тоже"])


EXPECTED = ("«При подключении MCP confluence-delta после 2-3 запросов получаю ошибку 401, токен при этом "
            "валидный. Из-за этого приходится каждый час перевыпускать токен вручную.»")


# ----------------------------------------------------------------------------- парсинг


@pytest.mark.parametrize("raw,expected", [
    ('{"quotes": [{"message": 2, "sentences": [3, 4]}]}', [(2, [3, 4])]),
    ('<think>сравню {1} и [3]</think>\n```json\n{"quotes": [{"message": 1, "sentences": [2]}]}\n```', [(1, [2])]),
    ('{"message": 3, "sentences": [1]}', [(3, [1])]),
    ('{"ranking": [2, 1]}', [(2, []), (1, [])]),      # старый формат — сообщения без предложений
    ("Лучшая цитата: 2.3 и 2.4", [(2, [3, 4])]),
])
def test_parse_selection(raw, expected):
    assert [(e.message, e.sentences) for e in parse_selection(raw).quotes] == expected


def test_parse_selection_garbage():
    with pytest.raises(ValueError):
        parse_selection("<think>думаю про 7 и 9")


def test_normalize_selection_bounds_and_dupes():
    shown = [["a", "b"], ["c"]]
    sel = Selection.model_validate({"quotes": [
        {"message": 1, "sentences": [2, 2, 9]}, {"message": 1, "sentences": [1]},
        {"message": 7, "sentences": [1]}, {"message": 2, "sentences": []}]})
    assert normalize_selection(sel, shown) == [[0, [1]], [1, []]]


def test_prefilter_drops_noise_and_does_not_penalize_long():
    c = prefilter(topic(), __import__("golos_digest.config").config.QuoteConfig(), 10)
    assert "+1" not in c and "у меня тоже" not in c
    assert c[0] == ENV_FIRST  # длинное письмо с сильным фрагментом не проигрывает из-за длины


# ----------------------------------------------------------------------------- селектор


def test_heuristic_skips_environment_and_greeting(cfg):
    q = QuoteSelector(cfg).select(topic())[0]
    assert q.text == EXPECTED


@pytest.mark.parametrize("mode,calls,cached", [
    ("tool", 1, True),     # structured output
    ("text", 2, True),     # structured вернул None -> текстовый фолбэк (с <think>)
    ("fail", 2, False),    # ретраи -> эвристика, сбой не кэшируется
])
def test_selector_modes(cfg, mode, calls, cached):
    cfg.llm.backend, cfg.llm.retries = "gigachat", 1
    sel = QuoteSelector(cfg, llm=FakeModel(mode=mode))
    q = sel.select(topic())[0]
    assert sel.llm.calls == calls
    assert q.text == EXPECTED       # и LLM (предл. 4-5), и эвристика выбирают проблему, а не окружение
    assert bool(sel.cache.data) is cached


def test_llm_sentence_choice_is_respected(cfg):
    """LLM выбрала одно предложение — цитата ровно из него, без добавок."""
    cfg.llm.backend = "gigachat"
    model = FakeModel(answer={"quotes": [{"message": 1, "sentences": [4]}]})
    q = QuoteSelector(cfg, llm=model).select(topic())[0]
    assert q.text.startswith("«При подключении") and q.text.endswith("валидный.»")


def test_llm_gap_filled_or_marked(cfg):
    cfg.llm.backend = "gigachat"
    model = FakeModel(answer={"quotes": [{"message": 1, "sentences": [5, 2]}]})  # 2 и 5: пропуск из двух
    q = QuoteSelector(cfg, llm=model).select(topic())[0]
    assert "… Из-за этого" in q.text  # несмежные предложения — через «…», а не склеены


def test_structured_output_off_single_call(cfg):
    cfg.llm.backend, cfg.llm.retries, cfg.llm.structured_output = "gigachat", 1, False
    sel = QuoteSelector(cfg, llm=FakeModel(mode="text"))
    assert sel.select(topic())[0].text == EXPECTED
    assert sel.llm.calls == 1  # без попытки function calling


def test_two_quotes_from_different_messages(cfg):
    cfg.llm.backend, cfg.quotes.max_quotes = "gigachat", 2
    model = FakeModel(answer={"quotes": [{"message": 1, "sentences": [4]}, {"message": 1, "sentences": [5]},
                                         {"message": 2, "sentences": [1]}]})
    qs = QuoteSelector(cfg, llm=model).select(topic())
    assert len(qs) == 2 and qs[1].text == f"«{SHORT}»"


# ----------------------------------------------------------------------------- фабрика модели


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
