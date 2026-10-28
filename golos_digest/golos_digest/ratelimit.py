"""Бережные вызовы LLM: общий на все потоки «шлагбаум» + классификация ошибок.

- не больше N одновременных запросов (N снижается при 429 и плавно восстанавливается);
- не чаще requests_per_minute (если задано);
- на 429 — общая пауза для ВСЕХ потоков: Retry-After от сервера или экспоненциальная (5→10→20…≤120 с);
- 5xx / таймауты / обрывы — повтор с паузой; 400/401/403/404/413/422 — без повторов.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from typing import Callable, TypeVar

log = logging.getLogger(__name__)
T = TypeVar("T")


# ----------------------------------------------------------------------------- классификация ошибок


def _status(e: BaseException) -> int | None:
    for obj in (e, getattr(e, "response", None)):
        code = getattr(obj, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def is_rate_limit(e: BaseException) -> bool:
    return _status(e) == 429 or "ratelimit" in type(e).__name__.lower()


def retry_after(e: BaseException) -> float | None:
    v = getattr(e, "retry_after", None)
    if isinstance(v, (int, float)) and v > 0:
        return float(v)
    for obj in (e, getattr(e, "response", None)):
        headers = getattr(obj, "headers", None)
        if headers:
            try:
                return max(float(headers.get("retry-after")), 0.0) or None
            except (TypeError, ValueError):
                pass
    return None


def is_fatal(e: BaseException) -> bool:
    """Ошибки, которые повтором не лечатся: авторизация, доступ, неверный запрос/модель."""
    return _status(e) in (400, 401, 403, 404, 413, 422) or type(e).__name__ in (
        "AuthenticationError", "ForbiddenError", "NotFoundError", "BadRequestError",
        "PermissionDeniedError", "ModelNotSpecifiedError")


def is_transient(e: BaseException) -> bool:
    code = _status(e)
    if code is not None:
        return code >= 500 or code in (408, 409)
    name = type(e).__name__.lower()
    return any(k in name for k in ("timeout", "connect", "remoteprotocol", "readerror", "network", "servererror"))


class FatalLLMError(RuntimeError):
    """Повторять бессмысленно (401/403/404/…): дальше по прогону LLM не вызываем."""


# ----------------------------------------------------------------------------- шлагбаум


class RateGate:
    def __init__(self, max_concurrency: int = 2, requests_per_minute: float = 0,
                 base_backoff: float = 5.0, max_backoff: float = 120.0, recover_after: int = 5,
                 transient_backoff: float = 2.0):
        self.max_concurrency = max(1, max_concurrency)
        self.limit = self.max_concurrency            # текущий лимит параллельности (AIMD)
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self.base_backoff, self.max_backoff = base_backoff, max_backoff
        self.recover_after = recover_after
        self.transient_backoff = transient_backoff
        self._cv = threading.Condition()
        self._active = 0
        self._pause_until = 0.0
        self._next_start = 0.0
        self._streak = 0          # подряд 429 (для экспоненты)
        self._ok_since_429 = 0
        self.n_429 = 0

    @property
    def paused_for(self) -> float:
        return max(0.0, self._pause_until - time.monotonic())

    def _acquire(self) -> None:
        with self._cv:
            while True:
                now = time.monotonic()
                wait = max(self._pause_until - now, self._next_start - now)
                if wait <= 0 and self._active < self.limit:
                    self._active += 1
                    self._next_start = now + self.min_interval
                    return
                self._cv.wait(timeout=wait if wait > 0 else None)

    def _release(self) -> None:
        with self._cv:
            self._active -= 1
            self._cv.notify_all()

    def _on_success(self) -> None:
        with self._cv:
            self._streak = 0
            self._ok_since_429 += 1
            if self.limit < self.max_concurrency and self._ok_since_429 >= self.recover_after:
                self.limit += 1
                self._ok_since_429 = 0
                self._cv.notify_all()

    def _on_429(self, server_hint: float | None) -> float:
        with self._cv:
            self.n_429 += 1
            self._streak += 1
            self._ok_since_429 = 0
            self.limit = max(1, self.limit // 2)
            delay = server_hint or min(self.base_backoff * 2 ** (self._streak - 1), self.max_backoff)
            delay *= random.uniform(1.0, 1.25)  # джиттер: потоки не просыпаются одной волной
            until = time.monotonic() + delay
            if until > self._pause_until:
                self._pause_until = until
                log.warning("429 Too Many Requests: пауза %.0f с для всех запросов, параллельность → %d",
                            delay, self.limit)
            self._cv.notify_all()
            return delay

    def call(self, fn: Callable[[], T], *, rate_limit_retries: int = 8, retries: int = 3) -> T:
        """Выполняет fn() с соблюдением лимитов. 429 не расходует обычные ретраи."""
        n429 = nerr = 0
        while True:
            self._acquire()
            try:
                result = fn()
            except Exception as e:
                self._release()
                if is_rate_limit(e):
                    n429 += 1
                    if n429 > rate_limit_retries:
                        raise
                    self._on_429(retry_after(e))
                    continue
                if is_fatal(e):
                    raise FatalLLMError(f"{type(e).__name__}: {e}") from e
                if is_transient(e) and nerr + 1 < retries:
                    nerr += 1
                    time.sleep(min(self.transient_backoff * 2 ** (nerr - 1), 30) * random.uniform(1.0, 1.3))
                    continue
                raise
            self._release()
            self._on_success()
            return result
