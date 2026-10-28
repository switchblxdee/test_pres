"""Конфигурация генератора. Все значения по умолчанию рабочие; YAML только переопределяет."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent

# Канонические имена колонок -> допустимые варианты заголовков (сравнение без регистра,
# пробелов по краям, с ё == е и схлопнутыми пробелами).
MARKUP_COLUMNS: dict[str, list[str]] = {
    "date": ["Дата сигнала от клиента", "Дата сигнала", "Дата"],
    "source": ["Источник сигнала", "Источник"],
    "product": ["Объект сигнала", "Продукт"],
    "topic": ["Кластер сигналов 2 уровня", "Кластер 2 уровня", "Тема"],
    "text": ["Текст из источника обратной связи", "Текст", "Текст сигнала"],
}

SUMMARY_COLUMNS: dict[str, list[str]] = {
    "product": ["Продукт"],
    "topic": ["Тема"],
    "cur": ["Сигналы текущей недели"],
    "prev": ["Сигналы прошлой недели"],
    "dyn_pct": ["Динамика %", "Динамика, %", "Динамика"],
    "status": ["Статус"],
    "dyn_abs": ["Динамика кол-во", "Динамика кол-ва", "Динамика количество"],
    "release_date": ["Дата выпуска"],
    "period": ["Период"],
    "number": ["Номер", "№"],
    "signals_total": ["Кол-во проанализированных сигналов"],
    "sources_total": ["Кол-во источников"],
    # Необязательные колонки: если появятся в своде, будут использованы.
    "status_icon": ["Иконка статуса", "Тип статуса", "Категория статуса"],
    "resumed": ["Возобновилась", "Возобновилось обсуждение", "Возобновление"],
}

SUMMARY_REQUIRED = ("product", "topic", "cur")
MARKUP_REQUIRED = ("product", "topic", "text")


@dataclass
class LLMConfig:
    # gigachat | openai | heuristic
    backend: str = "gigachat"
    model: str = "GigaChat-2-Max"
    # GigaChat
    credentials_env: str = "GIGACHAT_CREDENTIALS"
    scope: str = "GIGACHAT_API_CORP"
    base_url: str | None = None          # для openai-совместимого API или своего контура GigaChat
    api_key_env: str = "OPENAI_API_KEY"  # для backend=openai
    verify_ssl_certs: bool = False
    ca_bundle_file: str | None = None
    temperature: float = 0.05
    timeout: float = 90.0
    retries: int = 3
    max_workers: int = 8
    max_candidates: int = 40             # сколько сообщений по теме максимум уходит в промпт
    candidate_max_chars: int = 900       # обрезка длинных сообщений внутри промпта
    cache_dir: str = ".cache"


@dataclass
class QuoteConfig:
    max_quotes: int = 1          # 1 = одна лучшая цитата; 2 = добавить вторую, если влезает
    max_quote_lines: int = 5     # жёсткий потолок высоты цитаты на слайде
    ideal_min_chars: int = 60
    ideal_max_chars: int = 420
    min_chars: int = 20          # короче этого сообщение не рассматривается вовсе


@dataclass
class FontConfig:
    # Пути к TTF/OTF для измерения текста. Если не заданы — ищем через fc-match,
    # затем фолбэк. Для точных переносов поставь сюда реальные файлы SB Sans Display.
    regular: str | None = None
    semibold: str | None = None
    light: str | None = None
    fonts_dir: str | None = None       # папка, где искать SBSansDisplay-*.ttf/otf
    width_safety: float = 1.04         # запас по ширине строки (защита от переполнения)


@dataclass
class Config:
    template_path: str = str(PROJECT_DIR / "template" / "template.pptx")
    author: str = "Голос IT"           # автор в свойствах файла (в шаблоне были ФИО сотрудников)
    hide_zero_topics: bool = True      # темы с 0 сигналов на текущей неделе не выводим
    filter_markup_by_period: bool = True
    # Правила выбора иконки в карточке статуса, если в своде нет колонки "Иконка статуса".
    # Значение: pain (логотип, «В списке болей») | analysis (лупа, «На анализе»).
    status_icon_rules: list[list[str]] = field(default_factory=lambda: [
        [r"кандидат\s+в\s+бол|на\s+анализе", "analysis"],
        [r"в\s+списк\w*\s+бол", "pain"],
    ])
    resumed_status_regex: str = r"возобнов"
    llm: LLMConfig = field(default_factory=LLMConfig)
    quotes: QuoteConfig = field(default_factory=QuoteConfig)
    fonts: FontConfig = field(default_factory=FontConfig)


def _merge(dc: Any, data: dict) -> Any:
    known = {f.name: f for f in fields(dc)}
    for key, value in (data or {}).items():
        if key not in known:
            raise ValueError(f"Неизвестный параметр конфига: {key}")
        current = getattr(dc, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(dc, key, value)
    return dc


def load_config(path: str | os.PathLike | None) -> Config:
    cfg = Config()
    if path:
        import yaml

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        _merge(cfg, data)
    return cfg
