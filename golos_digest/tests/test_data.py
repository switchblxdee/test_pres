import pytest

from golos_digest.data import load_digest
from tests.conftest import markup_row, summary_row


def topics_by_title(d):
    return {t.title: t for t in d.topics}


def test_dynamics_new_resumed_and_status(make_xlsx, cfg):
    s = [
        summary_row("P", "Падение", 36, 39),
        summary_row("P", "Рост", 11, 6),
        summary_row("P", "Новая", 5, 0),
        summary_row("P", "Флаг возобновления", 4, 0, status="возобновилось"),
        summary_row("P", "Возобновилась со статусом", 4, 0, status="В работе у команды, обсуждение возобновилось"),
        summary_row("P", "Ноль", 0, 3),
        summary_row("P", "Без прошлой, но с процентом", 10, None, **{"Динамика %": "-20%"}),
        summary_row("P", "Без прошлой и без процента", 10, None),
        summary_row("P", "Через абсолютную динамику", 10, None, **{"Динамика кол-во": -10}),
    ]
    d = load_digest(make_xlsx(s, [markup_row("P", "Падение", "текст сообщения про падение сервиса")]), cfg)
    t = topics_by_title(d)
    assert t["Падение"].dyn_pct == -8 and not t["Падение"].is_new
    assert t["Рост"].dyn_pct == 83
    assert t["Новая"].is_new and t["Новая"].dyn_pct is None
    assert t["Флаг возобновления"].resumed and t["Флаг возобновления"].status == ""
    # статус с осмысленным текстом не должен стираться
    assert t["Возобновилась со статусом"].resumed
    assert t["Возобновилась со статусом"].status.startswith("В работе у команды")
    assert "Ноль" not in t  # hide_zero_topics
    assert t["Без прошлой, но с процентом"].dyn_pct == -20 and not t["Без прошлой, но с процентом"].is_new
    assert t["Без прошлой и без процента"].is_new
    assert t["Через абсолютную динамику"].prev == 20 and t["Через абсолютную динамику"].dyn_pct == -50
    assert d.meta.active_topics == 8 and d.meta.signals_total == 433 and d.meta.number == "20"


@pytest.mark.parametrize("status,icon", [
    ("ДЗО и Non-agile - в списке болей", "pain"),
    ("кандидат в боль - на анализе", "analysis"),
    ("Боль закрыта", None),          # не должно ложно срабатывать
    ("ограничение инфраструктуры", None),
])
def test_status_icons(make_xlsx, cfg, status, icon):
    d = load_digest(make_xlsx([summary_row("P", "T", 5, 4, status=status)], [markup_row("P", "T", "сообщение")]), cfg)
    assert d.topics[0].status_icon == icon


def test_header_offset_and_column_aliases(make_xlsx, cfg):
    path = make_xlsx([summary_row("P", "T", 5, 4)], [markup_row("P", "T", "сообщение")], summary_offset=2)
    d = load_digest(path, cfg)
    assert [t.title for t in d.topics] == ["T"]


def test_ordering_products_by_volume_topics_desc(make_xlsx, cfg):
    s = [summary_row("Small", "a", 3, 1), summary_row("Big", "b", 10, 5), summary_row("Big", "c", 20, 5)]
    d = load_digest(make_xlsx(s, [markup_row("Big", "b", "x" * 30)]), cfg)
    assert [p for p, _ in d.products] == ["Big", "Small"]
    assert [t.title for t in d.products[0][1]] == ["c", "b"]


def test_period_filter_and_fallback(make_xlsx, cfg):
    m = [markup_row("P", "T", "в периоде", day=12), markup_row("P", "T", "вне периода", day=1)]
    d = load_digest(make_xlsx([summary_row("P", "T", 5, 4)], m), cfg)
    assert d.topics[0].candidates == ["в периоде"]


def test_missing_required_column(make_xlsx, cfg, tmp_path):
    bad = [{"Продукт": "P", "Сигналы текущей недели": 1}]
    with pytest.raises(ValueError, match="Тема"):
        load_digest(make_xlsx(bad, [markup_row("P", "T", "x")]), cfg)


def test_no_topics_raises(make_xlsx, cfg):
    with pytest.raises(ValueError, match="нет ни одной темы"):
        load_digest(make_xlsx([summary_row("P", "T", 0, 4)], [markup_row("P", "T", "x")]), cfg)


def test_empty_markup_sheet(tmp_path, cfg):
    """Неделя без сообщений в разметке: дайджест строится, просто без цитат."""
    import pandas as pd
    from tests.conftest import summary_row
    p = tmp_path / "e.xlsx"
    cols = ["Дата сигнала от клиента", "Источник сигнала", "Объект сигнала",
            "Кластер сигналов 2 уровня", "Текст из источника обратной связи"]
    with pd.ExcelWriter(p, engine="openpyxl") as xw:
        pd.DataFrame(columns=cols).to_excel(xw, sheet_name="разметка", index=False)
        pd.DataFrame([summary_row("P", "T", 5, 4)]).to_excel(xw, sheet_name="свод", index=False)
    d = load_digest(p, cfg)
    assert d.topics[0].candidates == [] and d.meta.signals_total == 433
