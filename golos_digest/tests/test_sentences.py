import pytest

from golos_digest.sentences import Quote, best_quote, quote_from_ids, shorten_sentence, split_sentences


@pytest.mark.parametrize("text,expected", [
    ("Не работает. Помогите!", ["Не работает.", "Помогите!"]),
    ("Сломалось, т.е. совсем. И т. д. всё плохо.", ["Сломалось, т.е. совсем.", "И т. д. всё плохо."]),
    ("Версия 26.9.64 не стартует. см. лог", ["Версия 26.9.64 не стартует.", "см. лог"]),
    ("Ссылка https://x.sbrf.ru/a.b?c=1 не открывается. Почему?",
     ["Ссылка https://x.sbrf.ru/a.b?c=1 не открывается.", "Почему?"]),
    ("- Подскажите, у всех упало?\n- У меня аналогично.", ["Подскажите, у всех упало?", "У меня аналогично."]),
    ("Индексация идёт час\nи падает с OOM.\nПриходится перезапускать.",
     ["Индексация идёт час и падает с OOM.", "Приходится перезапускать."]),
    ("Ждём фикса... Очень мешает!", ["Ждём фикса...", "Очень мешает!"]),
    ("Иванов И. И. сказал, что починят.", ["Иванов И. И. сказал, что починят."]),
])
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected


@pytest.mark.parametrize("topic,text,must_have,must_not", [
    ("Сложности с MCP",
     "Добрый день! У меня рабочий стол на VDI, Windows 10, стоит VS Code 1.85. Я системный аналитик. "
     "При подключении MCP получаю ошибку 401. Из-за этого приходится перевыпускать токен вручную. Спасибо!",
     ["ошибку 401", "вручную"], ["VDI", "аналитик", "Спасибо", "Добрый"]),
    ("Сбой MCP confluence",
     "- Подскажите, у всех MCP confluence-delta не работает, или только у меня?\n- У меня аналогично.",
     ["не работает"], ["аналогично"]),
    ("Доступ к Desktop",
     "Коллеги, добрый день, при установке Desktop через SUS безопасность отказывает, нет роли разработчик.",
     ["отказывает"], ["Коллеги", "добрый день"]),
])
def test_best_quote_focuses_on_problem(topic, text, must_have, must_not):
    q, _ = best_quote(text, topic, 420)
    for s in must_have:
        assert s in q.text
    for s in must_not:
        assert s not in q.text


def test_greeting_prefix_stripped_and_capitalized():
    q, _ = best_quote("коллеги, добрый день, при установке через SUS безопасность отказывает.", "Доступ", 420)
    assert q.text == "«При установке через SUS безопасность отказывает.»"


def test_long_runon_sentence_cut_on_clause_boundaries():
    s = ("коллеги привет я аналитик у нас в команде все сидят на ssd и нам говорят переходить на gigacode "
         "desktop но при установке через sus безопасность отказывает потому что нет роли разработчик "
         "и что теперь делать непонятно")
    frag = shorten_sentence(s, "Доступ к Desktop", 100)
    assert frag.startswith("…") and frag.endswith("…")
    assert "безопасность отказывает" in frag and "нет роли разработчик" in frag
    assert len(frag) <= 102
    # слова не разрезаны: каждое слово фрагмента есть в исходнике
    assert all(w in s.split() for w in frag.strip("…").split())


def test_drop_edge_never_drops_most_important():
    q = Quote(["Контекст.", "Главное: всё упало.", "Мелочь."], [0.1, 5.0, 0.2])
    q1 = q.drop_edge()
    assert q1.sentences == ["Главное: всё упало.", "Мелочь."]
    q2 = q1.drop_edge()
    assert q2.sentences == ["Главное: всё упало."]
    assert q2.drop_edge() is None


def test_quote_from_ids_fills_single_gap_marks_long_gap():
    sents = ["Ошибка 500 при сохранении.", "Пробовал в Chrome.", "Спасибо!", "Приходится делать вручную."]
    q = quote_from_ids(sents, [0, 1], "Сохранение")
    assert q.text == "«Ошибка 500 при сохранении. Пробовал в Chrome.»"
    q = quote_from_ids(sents, [0, 2 + 1], "Сохранение")  # 0 и 3: пропуск из двух — «…»
    assert q.text == "«Ошибка 500 при сохранении. … Приходится делать вручную.»"
    q = quote_from_ids(["А сломалось.", "Пробовал перезапуск.", "Б не помогло."], [2, 0], "x")
    assert "Пробовал перезапуск." in q.text  # одиночный нейтральный пропуск заполняется
