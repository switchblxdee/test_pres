import threading
import time

import httpx
import pytest
from gigachat.exceptions import AuthenticationError, RateLimitError, ServerError

from golos_digest.data import Topic
from golos_digest.quotes import QuoteSelector
from golos_digest.ratelimit import FatalLLMError, RateGate
from tests.test_quotes import ENV_FIRST, SHORT, FakeModel


def _429(after="0.05"):
    return RateLimitError("https://ift/v1/chat", 429, b"Too Many Requests", httpx.Headers({"retry-after": after}))


def test_gate_waits_out_429_without_spending_retries():
    gate = RateGate(2, base_backoff=0.01)
    seq = iter([_429(), _429(), "ok"])

    def fn():
        x = next(seq)
        if isinstance(x, Exception):
            raise x
        return x

    t = time.monotonic()
    assert gate.call(fn, retries=1) == "ok"
    assert gate.n_429 == 2 and time.monotonic() - t >= 0.1   # Retry-After соблюдён
    assert gate.limit == 1                                    # параллельность снижена


def test_gate_fatal_no_retries():
    calls = []

    def fn():
        calls.append(1)
        raise AuthenticationError("https://ift/v1/chat", 401, b"bad cert", None)

    with pytest.raises(FatalLLMError):
        RateGate(2).call(fn)
    assert len(calls) == 1


def test_gate_retries_5xx_then_gives_up():
    calls = []

    def fn():
        calls.append(1)
        raise ServerError("https://ift/v1/chat", 503, b"busy", None)

    with pytest.raises(ServerError):
        RateGate(2, transient_backoff=0.01).call(fn, retries=3)
    assert len(calls) == 3


def test_gate_never_exceeds_concurrency():
    gate = RateGate(3)
    active = peak = 0
    lock = threading.Lock()

    def fn():
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1

    threads = [threading.Thread(target=gate.call, args=(fn,)) for _ in range(20)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert peak <= 3


def _topics(n):
    return [Topic("P", f"Ошибки авторизации MCP {i}", 5, 4, -1, "", None, False, False,
                  candidates=[ENV_FIRST + f" Случай {i}.", SHORT]) for i in range(n)]


def test_toolless_model_one_request_per_topic_after_first(cfg):
    cfg.llm.backend, cfg.llm.max_workers = "gigachat", 1
    model = FakeModel(mode="text")
    sel = QuoteSelector(cfg, llm=model)
    for t in _topics(5):
        assert sel.select(t)
    assert model.calls == 1 + 5   # одна пробная structured-попытка, дальше только текст
    assert sel.llm_failures == 0


def test_rate_limited_server_all_topics_succeed(cfg):
    """Сервер пускает 1 запрос одновременно, иначе 429 — все темы всё равно через LLM."""
    cfg.llm.backend, cfg.llm.max_workers = "gigachat", 4
    lock = threading.Lock()

    class Server(FakeModel):
        active: int = 0
        n429: int = 0

        def _generate(self, *a, **k):
            with lock:
                if self.active >= 1:
                    self.n429 += 1
                    raise _429("0.02")
                self.active += 1
            try:
                time.sleep(0.01)
                return super()._generate(*a, **k)
            finally:
                with lock:
                    self.active -= 1

    model = Server()
    sel = QuoteSelector(cfg, llm=model)
    sel.gate.base_backoff = 0.01
    ths = [threading.Thread(target=sel.select, args=(t,)) for t in _topics(8)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    assert sel.llm_calls == 8 and sel.llm_failures == 0
    assert model.n429 <= 12


def test_fatal_disables_llm_for_rest_of_run(cfg):
    cfg.llm.backend, cfg.llm.max_workers = "gigachat", 1

    class Bad(FakeModel):
        def _generate(self, *a, **k):
            self.calls += 1
            raise AuthenticationError("https://ift/v1/chat", 401, b"bad cert", None)

    model = Bad()
    sel = QuoteSelector(cfg, llm=model)
    for t in _topics(4):
        assert sel.select(t)          # цитаты есть — эвристикой
    assert model.calls == 1 and sel.llm_failures == 4
