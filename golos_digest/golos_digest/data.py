"""Чтение и нормализация xlsx: листы «разметка» и «свод»."""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .config import (
    MARKUP_COLUMNS,
    MARKUP_REQUIRED,
    SUMMARY_COLUMNS,
    SUMMARY_REQUIRED,
    Config,
)

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- модели


@dataclass
class Topic:
    product: str
    title: str
    cur: int
    prev: int | None
    dyn_pct: int | None          # None -> тема новая / возобновилась, бейджа нет
    status: str
    status_icon: str | None      # pain | analysis | None
    is_new: bool
    resumed: bool
    candidates: list[str] = field(default_factory=list)
    quotes: list[str] = field(default_factory=list)


@dataclass
class Meta:
    release_date: str
    period: str
    number: str
    signals_total: int
    sources_total: int
    active_topics: int = 0
    new_topics: int = 0


@dataclass
class Digest:
    meta: Meta
    products: list[tuple[str, list[Topic]]]

    @property
    def topics(self) -> list[Topic]:
        return [t for _, ts in self.products for t in ts]


# ----------------------------------------------------------------------------- утилиты


def norm(s: object) -> str:
    """Ключ для сравнения строк: регистр, ё/е, пробелы, кавычки."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s).replace("\xa0", " ").replace("ё", "е").replace("Ё", "Е")
    s = re.sub(r"[«»\"“”]", "", s)
    return re.sub(r"\s+", " ", s).strip().casefold()


def clean_text(s: object) -> str:
    if is_empty(s):
        return ""
    s = str(s).replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    return s.strip()


def is_empty(v: object) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(v, str) and not v.strip()


def to_int(v: object) -> int | None:
    if is_empty(v):
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return int(round(float(v)))
    s = re.sub(r"[\s\xa0]", "", str(v)).replace(",", ".").replace("−", "-")
    m = re.search(r"-?\d+(\.\d+)?", s)
    return int(round(float(m.group()))) if m else None


def to_float(v: object) -> float | None:
    if is_empty(v):
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    s = re.sub(r"[\s\xa0]", "", str(v)).replace(",", ".").replace("−", "-")
    m = re.search(r"[-+]?\d+(\.\d+)?", s)
    return float(m.group()) if m else None


def to_bool(v: object) -> bool:
    if is_empty(v):
        return False
    if isinstance(v, (int, float)):
        return bool(v)
    return norm(v) in {"1", "да", "true", "yes", "y", "+", "истина", "возобновилась", "возобновилось"}


def to_date(v: object) -> date | None:
    if is_empty(v):
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, pd.Timestamp):
        return v.date()
    if isinstance(v, (int, float)):  # excel serial
        try:
            return (pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(v))).date()
        except (ValueError, OverflowError):
            return None
    try:
        ts = pd.to_datetime(str(v).strip(), dayfirst=True, errors="coerce")
    except (ValueError, TypeError):
        return None
    return None if pd.isna(ts) else ts.date()


def first_value(series: pd.Series | None) -> object:
    if series is None:
        return None
    for v in series:
        if not is_empty(v):
            return v
    return None


# ----------------------------------------------------------------------------- чтение листов


def _find_sheet(sheets: dict[str, pd.DataFrame], stem: str) -> pd.DataFrame:
    for name, df in sheets.items():
        if norm(name).startswith(stem):
            return df
    raise ValueError(f"Не найден лист '{stem}…'. Листы в файле: {list(sheets)}")


def _header_score(values, spec: dict[str, list[str]]) -> int:
    names = {norm(v) for v in values}
    return sum(any(norm(a) in names for a in aliases) for aliases in spec.values())


def _promote_header(df: pd.DataFrame, spec: dict[str, list[str]], scan_rows: int = 15) -> pd.DataFrame:
    """Шапка не обязательно в первой строке (над таблицей бывает заголовок отчёта):
    берём ту из первых строк, где совпало больше всего известных колонок."""
    best_row, best = -1, _header_score(df.columns, spec)
    for i in range(min(scan_rows, len(df))):
        sc = _header_score(df.iloc[i].tolist(), spec)
        if sc > best:
            best_row, best = i, sc
    if best_row >= 0:
        header = [str(v).strip() if not is_empty(v) else f"_col{j}" for j, v in enumerate(df.iloc[best_row])]
        df = df.iloc[best_row + 1:].copy()
        df.columns = header
    # дубли имён колонок: оставляем первую
    return df.loc[:, ~pd.Index([norm(c) for c in df.columns]).duplicated()]


def _resolve_columns(df: pd.DataFrame, spec: dict[str, list[str]], required, sheet: str) -> pd.DataFrame:
    df = _promote_header(df, spec)
    df = df.dropna(how="all")
    by_norm = {norm(c): c for c in df.columns}
    mapping: dict[str, str] = {}
    for canon, aliases in spec.items():
        for alias in aliases:
            if norm(alias) in by_norm:
                mapping[canon] = by_norm[norm(alias)]
                break
    missing = [c for c in required if c not in mapping]
    if missing:
        need = {c: spec[c][0] for c in missing}
        raise ValueError(f"Лист '{sheet}': нет обязательных колонок {need}. Есть: {list(df.columns)}")
    out = pd.DataFrame({canon: df[col] for canon, col in mapping.items()})
    for canon in spec:
        if canon not in out:
            out[canon] = None
    return out


def _parse_period(period: str, year: int | None) -> tuple[date, date] | None:
    """'09.10 – 15.10' / '09.10.2026-15.10.2026' -> (date, date)."""
    parts = re.findall(r"(\d{1,2})\.(\d{1,2})(?:\.(\d{2,4}))?", period or "")
    if len(parts) < 2:
        return None
    try:
        res = []
        for d, m, y in parts[:2]:
            yy = int(y) if y else year
            if yy is None:
                return None
            if yy < 100:
                yy += 2000
            res.append(date(yy, int(m), int(d)))
        start, end = res
        if start > end:  # переход через Новый год: 29.12 – 04.01
            start = start.replace(year=start.year - 1)
        return start, end
    except ValueError:
        return None


def _fmt_date(v: object) -> str:
    d = to_date(v)
    return d.strftime("%d.%m.%Y") if d else ("" if is_empty(v) else str(v).strip())


def _fmt_period(v: object) -> str:
    if is_empty(v):
        return ""
    if isinstance(v, (datetime, date, pd.Timestamp)):  # в ячейке дата начала периода
        return f"{to_date(v):%d.%m}"
    s = str(v).strip()
    s = re.sub(r"^\s*период\s*:?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*[-–—]\s*", " – ", s)
    return s


def _status_icon(status: str, explicit: object, cfg: Config) -> str | None:
    if not is_empty(explicit):
        e = norm(explicit)
        if e in {"pain", "боль", "в списке болей"} or "списке бол" in e or e == "логотип":
            return "pain"
        if e in {"analysis", "анализ"} or "анализ" in e or "кандидат" in e:
            return "analysis"
        return None
    for pattern, icon in cfg.status_icon_rules:
        if re.search(pattern, status or "", flags=re.I):
            return icon
    return None


# ----------------------------------------------------------------------------- основное


def load_digest(path: str | Path, cfg: Config) -> Digest:
    sheets = pd.read_excel(path, sheet_name=None, dtype=object, engine="openpyxl")
    markup = _resolve_columns(_find_sheet(sheets, "размет"), MARKUP_COLUMNS, MARKUP_REQUIRED, "разметка")
    summary = _resolve_columns(_find_sheet(sheets, "свод"), SUMMARY_COLUMNS, SUMMARY_REQUIRED, "свод")

    # явный bool + .loc: на пустом листе map() даёт object-Series, и df[mask] выбрал бы колонки
    empty_s = summary["product"].map(is_empty).astype(bool) | summary["topic"].map(is_empty).astype(bool)
    summary = summary.loc[~empty_s].copy()
    markup = markup.loc[~markup["text"].map(is_empty).astype(bool)].copy()
    markup["text"] = markup["text"].map(clean_text)
    markup["_date"] = markup["date"].map(to_date)
    markup["_key"] = markup["product"].map(norm) + "\x1f" + markup["topic"].map(norm)
    markup["_topic"] = markup["topic"].map(norm)

    # ---- мета
    release_raw = first_value(summary["release_date"])
    release = to_date(release_raw)
    period = _fmt_period(first_value(summary["period"]))
    span = _parse_period(period, release.year if release else None)

    in_period = markup
    if cfg.filter_markup_by_period and span and markup["_date"].notna().any():
        mask = markup["_date"].map(lambda d: d is not None and span[0] <= d <= span[1]).astype(bool)
        if mask.any():
            in_period = markup.loc[mask]
        else:
            log.warning("Ни одна дата в разметке не попала в период %s — фильтр по датам отключён", period)

    if not period and in_period["_date"].notna().any():
        dates = in_period["_date"].dropna()
        period = f"{min(dates):%d.%m} – {max(dates):%d.%m}"

    number = to_int(first_value(summary["number"]))
    signals_total = to_int(first_value(summary["signals_total"]))
    sources_total = to_int(first_value(summary["sources_total"]))
    meta = Meta(
        release_date=_fmt_date(release_raw) or date.today().strftime("%d.%m.%Y"),
        period=period,
        number=str(number) if number is not None else "",
        signals_total=signals_total if signals_total is not None else len(in_period),
        sources_total=sources_total if sources_total is not None else int(in_period["source"].map(norm).replace("", pd.NA).nunique()),
    )

    # ---- кандидаты в цитаты: группировка один раз, а не фильтр на каждую тему
    by_key = {k: g["text"].tolist() for k, g in in_period.groupby("_key", sort=False)}
    by_topic = {k: g["text"].tolist() for k, g in in_period.groupby("_topic", sort=False)}
    all_by_key = {k: g["text"].tolist() for k, g in markup.groupby("_key", sort=False)}

    # ---- темы
    topics: list[Topic] = []
    seen: set[str] = set()
    for row in summary.itertuples(index=False):
        product = clean_text(row.product)
        title = clean_text(row.topic)
        key = norm(product) + "\x1f" + norm(title)
        if key in seen:
            log.warning("Дубль темы в своде, пропускаю: %s / %s", product, title)
            continue
        seen.add(key)

        cur = max(to_int(row.cur) or 0, 0)
        prev = to_int(row.prev)
        dyn_abs = to_int(row.dyn_abs)
        if prev is None and dyn_abs is not None:
            prev = cur - dyn_abs
        if prev is not None and prev < 0:
            log.warning("Отрицательные сигналы прошлой недели (%s) у темы «%s» — считаю 0", prev, title)
            prev = 0
        if cfg.hide_zero_topics and cur <= 0:
            continue

        status = clean_text(row.status)
        resumed = to_bool(row.resumed) or bool(re.search(cfg.resumed_status_regex, status, re.I))
        if resumed and not re.sub(rf"\w*{cfg.resumed_status_regex}\w*", "", status, flags=re.I).strip(" .,;:-–—!"):
            status = ""  # статус состоял только из флага «возобновилась» — в карточку не выводим

        raw_pct = None if is_empty(row.dyn_pct) else to_float(row.dyn_pct)
        # новая: прошлая неделя = 0, либо прошлая неизвестна и процента динамики нет совсем
        is_new = not resumed and (prev == 0 or (prev is None and raw_pct is None))

        dyn: int | None = None
        if not (is_new or resumed):
            if prev:
                dyn = round((cur - prev) / prev * 100)
            elif raw_pct is not None:
                # 0.08 из ячейки с процентным форматом = 8%; «-8%» строкой = -8
                frac = abs(raw_pct) <= 1 and "%" not in str(row.dyn_pct) and not float(raw_pct).is_integer()
                dyn = round(raw_pct * 100) if frac else round(raw_pct)

        candidates = by_key.get(key) or all_by_key.get(key) or by_topic.get(norm(title)) or []
        if not candidates:
            log.warning("Нет сообщений в разметке для темы: %s / %s", product, title)

        topics.append(Topic(
            product=product, title=title, cur=cur, prev=prev, dyn_pct=dyn,
            status=status, status_icon=_status_icon(status, row.status_icon, cfg),
            is_new=is_new, resumed=resumed, candidates=candidates,
        ))

    if not topics:
        raise ValueError("В своде нет ни одной темы с сигналами на текущей неделе")

    # продукты — по сумме сигналов, темы внутри — по убыванию сигналов (стабильная сортировка)
    groups: dict[str, list[Topic]] = {}
    for t in topics:
        groups.setdefault(t.product, []).append(t)
    products = sorted(groups.items(), key=lambda kv: -sum(t.cur for t in kv[1]))
    products = [(p, sorted(ts, key=lambda t: -t.cur)) for p, ts in products]

    meta.active_topics = len(topics)
    meta.new_topics = sum(t.is_new for t in topics)
    return Digest(meta=meta, products=products)
