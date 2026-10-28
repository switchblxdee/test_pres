import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from golos_digest.config import Config  # noqa: E402


def summary_row(product, topic, cur, prev, status="", **extra):
    row = {
        "Продукт": product, "Тема": topic, "Сигналы текущей недели": cur, "Сигналы прошлой недели": prev,
        "Динамика %": "", "Статус": status, "Динамика кол-во": "",
        "Дата выпуска": date(2026, 10, 16), "Период": "09.10 – 15.10", "Номер": 20,
        "Кол-во проанализированных сигналов": 433, "Кол-во источников": 16,
    }
    row.update(extra)
    return row


def markup_row(product, topic, text, day=10, source="Канал 1"):
    return {"Дата сигнала от клиента": date(2026, 10, day), "Источник сигнала": source,
            "Объект сигнала": product, "Кластер сигналов 2 уровня": topic,
            "Текст из источника обратной связи": text}


@pytest.fixture
def make_xlsx(tmp_path):
    def _make(summary, markup, name="in.xlsx", summary_offset=0):
        path = tmp_path / name
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            pd.DataFrame(markup).to_excel(xw, sheet_name="разметка", index=False)
            pd.DataFrame(summary).to_excel(xw, sheet_name="свод", index=False, startrow=summary_offset)
            if summary_offset:  # заголовок отчёта над таблицей
                xw.sheets["свод"].cell(row=1, column=1, value="Свод за неделю")
        return path
    return _make


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.llm.backend = "heuristic"
    c.llm.cache_dir = str(tmp_path / "cache")
    return c
