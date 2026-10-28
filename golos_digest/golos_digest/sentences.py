"""Работа с текстом обращений на уровне предложений.

- split_sentences: разбиение на предложения с учётом сокращений, версий, URL, диалоговых реплик;
- score_sentence: насколько предложение описывает проблему пользователя (а не приветствие,
  благодарность или описание окружения);
- best_window: лучший непрерывный фрагмент сообщения в пределах бюджета символов;
- Quote: цитата из целых предложений, которая ужимается только целыми предложениями с краёв.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

SPLITTER_VERSION = "2"  # входит в ключ кэша: поменялась логика разбиения — кэш пересчитается

# ----------------------------------------------------------------------------- разбиение

_ABBR = {
    "т.е", "т.к", "т.д", "т.п", "т.н", "т.ч", "др", "пр", "см", "напр", "ул", "г", "гг", "им", "руб",
    "тыс", "млн", "млрд", "стр", "рис", "ср", "англ", "прим", "мин", "сек", "e.g", "i.e", "etc", "vs",
}
_END = re.compile(r"([.!?…]+)([\"»”')\]]*)(?=\s|$)")
_BULLET = re.compile(r"^\s*(?:[-–—•*]|\d+[.)])\s+")
_LINE_END_PUNCT = re.compile(r"[.!?…:;»\"”)]\s*$")


def _logical_lines(text: str) -> list[str]:
    """Склеивает строки, разорванные посреди предложения (вставка из почты/мессенджера)."""
    out: list[str] = []
    for raw in text.replace("\r", "\n").split("\n"):
        line = raw.strip()
        if not line:
            continue
        starts_lower = line[0].islower() and not _BULLET.match(raw)
        if out and starts_lower and not _LINE_END_PUNCT.search(out[-1]):
            out[-1] = f"{out[-1]} {line}"
        else:
            out.append(line)
    return out


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for line in _logical_lines(text or ""):
        line = _BULLET.sub("", line)
        start = 0
        for m in _END.finditer(line):
            if m.group(1) == ".":  # одиночная точка — единственный неоднозначный случай
                tok = re.search(r"([\w.]+)$", line[start:m.start()])
                t = tok.group(1).lower().strip(".") if tok else ""
                if t in _ABBR or (len(t) == 1 and t.isalpha()):
                    continue  # «т.е.», «и т. д.», инициалы
            seg = line[start:m.end()].strip()
            if seg:
                sentences.append(seg)
            start = m.end()
        tail = line[start:].strip()
        if tail:
            sentences.append(tail)
    # обрывки вроде «...» или «)» приклеиваем к предыдущему предложению
    merged: list[str] = []
    for s in sentences:
        if merged and not re.search(r"\w", s):
            merged[-1] = f"{merged[-1]} {s}"
        else:
            merged.append(s)
    return merged


# ----------------------------------------------------------------------------- оценка

_PROBLEM = re.compile(
    r"не\s+работа|не\s+могу|не\s+мож|не\s+уда[её]т|не\s+получа|ошибк|error|exception|fail|"
    r"сбо[йияе]|пада[ею]т|упал|вылета|crash|завис|тормоз|долго|медленн|таймаут|timeout|"
    r"\b[45]\d\d\b|нет\s+доступ|доступа?\s+нет|отказ|запрещ|не\s+подключ|не\s+открыва|не\s+запуска|"
    r"не\s+отобража|не\s+видит|не\s+грузит|не\s+загружа|не\s+сохраня|не\s+срабатыва|пропа[лд]|"
    r"перестал|отвалива|погиба|невозможно|нельзя|проблем|баг\b|глюч|некорректн|неправильн|"
    r"обрыва|обреза|теря[ею]т|не\s+хватает|отсутству|нет\s+возможност",
    re.I,
)
_ASK = re.compile(
    r"когда\s+(будет|смогут|появ|почин|исправ)|как\s+(сделать|получить|подключить|настроить|исправить|"
    r"починить|обойти)|почему|что\s+делать|в\s+ч[её]м\s+(может\s+быть\s+)?(дело|проблема)|"
    r"у\s+всех\s+.*\?|или\s+только\s+у\s+меня",
    re.I,
)
_IMPACT = re.compile(
    r"приходится|из-за|в\s+результате|поэтому|блокир|не\s+можем|работать\s+невозможно|теря(ем|ется)|"
    r"вручную|простой|срыва|сроки|мешает|экономии\s+.*нет",
    re.I,
)
_REASON = re.compile(r"потому\s+что|так\s+как|поскольку|из-за|по\s+причине|нет\s+(?:роли|прав|доступа|возможност)", re.I)
_LEAD_CONJ = re.compile(r"^(?:и|но|а|однако|причем|причём|при\s+этом)\s+", re.I)
_GREET_WORD = r"(?:коллеги|всем|привет(?:ствую)?|добрый\s+(?:день|вечер)|доброе\s+утро|здравствуйте|друзья|команда)"
_GREETING = re.compile(rf"^{_GREET_WORD}(?:[\s,!.]+{_GREET_WORD})*[\s,!.]*", re.I)
_THANKS = re.compile(r"спасибо|благодар|заранее\s+спас", re.I)
_CONTEXT = re.compile(
    r"\bу\s+меня\s+(стоит|установлен|windows|win|mac|linux|ubuntu|ноутбук|комп)|\bверси[яиюей]\b|"
    r"\bос\b|\bwindows\b|\bmacos\b|\bubuntu\b|использую|работаю\s+(в|на|с|через)|"
    r"я\s+(системн|бизнес|аналитик|разработчик|тестировщик|менеджер|эксперт|руководител|инженер)|"
    r"трудоустроен|совместител|рабоч(ий|ем|его)\s+(стол|мест)|\bvdi\b|виртуальн|по\s+инструкции|"
    r"для\s+контекста|расскажу|опишу|ситуация\s+такая",
    re.I,
)
_NOISE_ONLY = re.compile(
    r"^\W*(\+\s*1|у\s+меня\s+(тоже|аналогично|так\s*же)|аналогично|та\s+же\s+(проблема|ситуация)|"
    r"спасибо|подскажите|помогите|актуально|up|ап)\W*$",
    re.I,
)
_URL = re.compile(r"https?://\S+")


def _stems(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-zа-яё0-9-]{4,}", text.lower())}


def score_sentence(s: str, topic: str = "", context_bonus: bool = True) -> float:
    """> 0 — описывает проблему/последствия/просьбу; ≈ 0 — нейтральный контекст;
    < 0 — приветствие, благодарность, «+1», окружение без проблемы, логи."""
    if _NOISE_ONLY.match(s):
        return -3.0
    score = 0.0
    problems = len(_PROBLEM.findall(s))
    score += min(problems, 3) * 2.0
    if _ASK.search(s):
        score += 1.5
    if _IMPACT.search(s):
        score += 1.5
    if _REASON.search(s):
        score += 1.0
    if topic:
        score += min(len(_stems(s) & _stems(topic)), 3) * 0.7
    if "?" in s:
        score += 0.3
    g = _GREETING.match(s)
    if g and len(s) - g.end() < 15:
        score -= 3.0  # предложение — только приветствие
    if _THANKS.search(s) and not problems:
        score -= 3.0
    is_context = bool(_CONTEXT.search(s)) and not problems
    letters = sum(c.isalpha() for c in s)
    if letters / max(len(s), 1) < 0.55:  # логи, код, стектрейсы
        score -= 2.5
    score -= 0.5 * len(_URL.findall(s))
    if len(s) < 12:
        score -= 1.0
    elif context_bonus and score >= 0 and len(s) >= 25 and not is_context:
        score += 0.2  # содержательное нейтральное предложение — полезный контекст рядом с проблемой
    if is_context:
        # окружение/должность/предыстория без признаков проблемы: совпадение слов с темой
        # (версия модели, название продукта) не должно вытягивать его в цитату
        score = min(score - 1.5, -1.0)
    return score


# ----------------------------------------------------------------------------- цитата


@dataclass
class Quote:
    """Цитата из целых предложений одного сообщения (в исходном порядке)."""
    sentences: list[str]
    weights: list[float]                      # важность каждого предложения
    gaps: list[bool] = field(default_factory=list)  # gaps[i]: перед sentences[i] пропущен текст

    def __post_init__(self) -> None:
        if not self.gaps:
            self.gaps = [False] * len(self.sentences)

    @classmethod
    def from_text(cls, text: str, topic: str = "") -> "Quote":
        sents = split_sentences(text) or [text.strip()]
        return cls(sents, [score_sentence(s, topic) for s in sents])

    @property
    def body(self) -> str:
        parts: list[str] = []
        for i, s in enumerate(self.sentences):
            if i == 0:
                g = _GREETING.match(s)
                if g and len(s) - g.end() >= 15:  # «Коллеги, добрый день, не работает X» -> «Не работает X»
                    s = s[g.end():]
                if s[:1].islower() and s[:1].isalpha():
                    s = s[:1].upper() + s[1:]
                parts.append(s)
            else:
                parts.append(("… " if self.gaps[i] and not s.startswith("…") else "") + s)
        return " ".join(parts)

    @property
    def text(self) -> str:
        return f"«{self.body}»"

    def __len__(self) -> int:
        return len(self.body)

    def without_noise(self) -> "Quote":
        """Срезает с краёв предложения с отрицательной важностью (приветствия, «спасибо»,
        окружение без проблемы); хотя бы одно предложение остаётся всегда."""
        i, j = 0, len(self.sentences)
        while j - i > 1 and self.weights[i] < 0:
            i += 1
        while j - i > 1 and self.weights[j - 1] < 0:
            j -= 1
        if (i, j) == (0, len(self.sentences)):
            return self
        return Quote(self.sentences[i:j], self.weights[i:j], [False] + self.gaps[i + 1:j])

    def drop_edge(self) -> "Quote | None":
        """Убирает менее важное из крайних предложений; самое важное не трогает никогда."""
        if len(self.sentences) <= 1:
            return None
        keep = max(range(len(self.weights)), key=lambda i: (self.weights[i], -i))
        if keep == 0:
            drop_first = False
        elif keep == len(self.sentences) - 1:
            drop_first = True
        else:
            drop_first = self.weights[0] < self.weights[-1]
        if drop_first:
            gaps = [False] + self.gaps[2:]
            return Quote(self.sentences[1:], self.weights[1:], gaps)
        return Quote(self.sentences[:-1], self.weights[:-1], self.gaps[:-1])


def best_window(sentences: list[str], scores: list[float], budget: int) -> tuple[int, int, float]:
    """Непрерывный фрагмент [i, j) с максимальной суммой оценок в пределах budget символов.
    При равенстве — короче. Одиночное предложение допускается даже сверх бюджета."""
    best = (0, 1, float("-inf"), 0)
    n = len(sentences)
    for i in range(n):
        total, acc = 0, 0.0
        for j in range(i, n):
            total += len(sentences[j]) + (1 if j > i else 0)
            if total > budget and j > i:
                break
            acc += scores[j]
            if (acc, -total) > (best[2], -best[3]):
                best = (i, j + 1, acc, total)
    return best[0], best[1], best[2]


# ----------------------------------------------------------------------------- длинные предложения

_CLAUSE_BREAK = re.compile(
    r"(?<=[,;:])\s+|\s+(?=[—–]\s)|"
    r"\s+(?=(?:но|а|однако|потому\s+что|поэтому|так\s+как|из-за\s+(?:этого|чего)|при\s+этом|"
    r"в\s+результате|причем|причём|хотя|и\s+при\s+этом|и\s+теперь|и\s+что)\b)",
    re.I,
)


def split_clauses(sentence: str) -> list[str]:
    parts = [p for p in _CLAUSE_BREAK.split(sentence) if p and p.strip()]
    return parts or [sentence]


def shorten_sentence(sentence: str, topic: str, budget: int) -> str:
    """Предложение длиннее бюджета -> лучший фрагмент из целых частей (по запятым и союзам)
    с «…» на месте выброшенного. Не обрезает слова — режет только на границах частей."""
    if len(sentence) <= budget:
        return sentence
    clauses = split_clauses(sentence)
    if len(clauses) == 1:
        return sentence  # резать не по чему — пусть решает вёрстка
    scores = [score_sentence(c, topic, context_bonus=False) for c in clauses]
    i, j, _ = best_window(clauses, scores, budget)
    text = " ".join(clauses[i:j]).strip().rstrip(",;:—–")
    if i > 0:
        text = "…" + _LEAD_CONJ.sub("", text)
    if j < len(clauses):
        text = text.rstrip(".!?…") + "…"
    return text


def _shorten_all(sents: list[str], topic: str, budget: int) -> list[str]:
    return [shorten_sentence(s, topic, budget) for s in sents]


def best_quote(text: str, topic: str, budget: int) -> tuple[Quote, float]:
    sents = _shorten_all(split_sentences(text) or [text.strip()], topic, budget)
    scores = [score_sentence(s, topic) for s in sents]
    i, j, value = best_window(sents, scores, budget)
    return Quote(sents[i:j], scores[i:j]), value


def quote_from_ids(sents: list[str], ids: list[int], topic: str, budget: int = 10**9) -> Quote | None:
    """Цитата из выбранных LLM предложений (0-based, порядок = важность).
    Одиночный пропуск между выбранными заполняется, если пропущенное не мусор; иначе «…»."""
    order = list(dict.fromkeys(i for i in ids if 0 <= i < len(sents)))
    if not order:
        return None
    rank = {i: len(order) - r for r, i in enumerate(order)}  # важность из порядка LLM
    chosen = sorted(order)
    filled: list[int] = [chosen[0]]
    for i in chosen[1:]:
        if i - filled[-1] == 2 and score_sentence(sents[i - 1], topic) >= 0:
            filled.append(i - 1)
        filled.append(i)
    weights = [float(rank.get(i, 0.1)) for i in filled]
    gaps = [False] + [b - a > 1 for a, b in zip(filled, filled[1:])]
    return Quote(_shorten_all([sents[i] for i in filled], topic, budget), weights, gaps)
