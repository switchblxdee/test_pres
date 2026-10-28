"""Вёрстка и сборка .pptx из шаблона-примера.

Подход: все визуальные компоненты (карточки, шапки разделов, бейджи, иконки, легенда)
берутся из эталонного слайда шаблона и клонируются как XML. Поэтому фон, шрифты,
скругления, векторные иконки и цвета совпадают с примером один в один, а код
отвечает только за данные и координаты.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pptx import Presentation

from .config import Config
from .data import Digest, Topic
from .measure import EMU_PER_INCH, Measurer
from .ooxml import (
    SlideBuilder, clone, ensure_text_color, find_by_id, get_box, harden_nowrap, recolor, run_styles, set_body_attr, set_box,
    set_fill, set_line, set_text, write_paragraphs,
)

log = logging.getLogger(__name__)
IN = EMU_PER_INCH

# ----------------------------------------------------------------------------- id фигур эталонного слайда (слайд 1 примера)

T = {
    "title": 42, "date": 7, "slide_num": 3,
    "top_block": (19, 38, 37, 36, 35, 80, 81, 84, 93),
    "kpi_signals": 20, "kpi_sources": 24, "kpi_active": 28, "kpi_new": 32,
    "kpi_card_label": ((10, 21), (23, 25), (27, 29), (31, 33)),   # (фон карточки, подпись под числом)
    "legend": (4, 6, 8, 11, 39, 77, 56, 57, 58, 59),
    "card": 50, "header": 49, "header_product": 48,
    "num": 51, "row_title": 52, "quote": 53, "count": 55, "badge": 83,
    "status_box": 61, "status_text": 62,
    "icon_pain": 6, "icon_analysis": 11, "icon_resumed": 57, "icon_new": 59,
}

# ----------------------------------------------------------------------------- сетка (EMU), снята с примера и выровнена

CONTENT_X, CONTENT_W = 334962, 11522074
TOP_FIRST = 1819350            # под KPI-блоком
TOP_OTHER = int(0.89 * IN)
BOTTOM = int(6.80 * IN)        # над легендой (легенда на 6.99")
ROW_OFFSET = 495506            # от верха карточки до первой строки
ROW_GAP = int(0.22 * IN)
CARD_PAD = int(0.15 * IN)
CARD_GAP = int(0.17 * IN)

NUM_X, NUM_SIZE = 501683, 360000
TITLE_X, TITLE_W = 982766, 6755844
TITLE_GAP = int(0.10 * IN)
QUOTE_PARA_GAP_PT = 6
COL_CX = 8884923               # центр колонки «Динамика»
COUNT_H = 215444
# Ширины однострочных боксов с большим запасом: некоторые просмотрщики без SB Sans
# подставляют более широкий шрифт и игнорируют wrap="none" — текст в тесном боксе
# переносится и наезжает на соседей. Выравнивание сохраняет визуальное положение.
COUNT_BOX_W = int(1.9 * IN)       # центр = COL_CX
KPI_NUM_BOX_W = int(1.3 * IN)     # левый край как в шаблоне
LABEL_INS = 91440                 # внутренний отступ текстбокса по умолчанию (0.1")
KPI_LABEL_RIGHT_GAP = 45720       # подпись KPI не доходит до края карточки на 0.05"
DATE_BOX_W = int(3.0 * IN)        # правый край как в шаблоне
PRODUCT_BOX_W = int(6.3 * IN)     # левый край как в шаблоне, до колонки «Динамика»
BADGE_DY, BADGE_H, BADGE_PAD = 250268, 310889, 108000
ICON = 216000
ICON_DY = 300000
STATUS_X, STATUS_W, STATUS_MIN_H = 9949428, 1800000, 561157
STATUS_TXT_ICON_X, STATUS_TXT_ICON_W = 9987687, 1271525
STATUS_ICON_X = 11325912
STATUS_INS_X, STATUS_INS_Y = 91440, 45720
RIGHT_MIN_H = BADGE_DY + BADGE_H

TITLE_FONT, TITLE_PT, TITLE_LH = "SB Sans Display Semibold", 12, int(12 * 1.21 * 12700)
QUOTE_FONT, QUOTE_PT, QUOTE_LH = "SB Sans Display Light", 10, int(10 * 1.21 * 12700)
STATUS_FONT, STATUS_PT = "SB Sans Display Light", 10
TITLE_MAX_LINES = 3
STATUS_MAX_LINES = 6
STATUS_PAD = int(0.1 * IN)
MIN_SQUEEZE_LINES = 2          # ниже этого цитату не ужимаем ради заполнения слайда
MIN_GLOBAL_QUOTE_LINES = 3     # общий лимит цитат ради экономии слайда — не ниже этого

STATUS_TEXT_COLOR = "000000"
RED_ARROW, RED_TEXT, RED_FILL = "FF0000", "E5383B", "FEE4E4"


def fmt_int(v: int) -> str:
    """12345 -> «12 345» с неразрывным пробелом (разряды не разрываются переносом)."""
    return f"{int(v):,}".replace(",", "\u00a0")


# ----------------------------------------------------------------------------- план вёрстки


@dataclass
class RowPlan:
    index: int
    topic: Topic
    top: int
    title: str
    title_h: int
    quotes: list[str]
    quote_h: int
    status: str
    box_h: int            # высота карточки статуса == высота строки (единый источник правды)

    @property
    def text_h(self) -> int:
        return self.title_h + (TITLE_GAP + self.quote_h if self.quotes else 0)

    @property
    def height(self) -> int:
        return max(self.box_h, RIGHT_MIN_H)


@dataclass
class SectionPlan:
    product: str
    top: int
    bottom: int = 0
    rows: list[RowPlan] = field(default_factory=list)


@dataclass
class SlidePlan:
    first: bool
    sections: list[SectionPlan] = field(default_factory=list)


@dataclass
class _RowBase:
    """Части строки, не зависящие от лимита строк цитаты, — меряются один раз."""
    title: str
    title_h: int
    status: str
    status_h: int


class Layouter:
    def __init__(self, cfg: Config, measurer: Measurer):
        self.cfg = cfg
        self.m = measurer
        # любая строка обязана помещаться на самый тесный слайд — первый, под KPI
        self.max_row_h = BOTTOM - TOP_FIRST - ROW_OFFSET - CARD_PAD
        self._bases: dict[int, _RowBase] = {}  # название/статус меряются один раз на все прогоны плана

    # ---- текст

    def _clip(self, text: str, font: str, pt: float, width: int, max_lines: int,
              opener: str = "", closer: str = "") -> str:
        """Обрезает по словам с «…», чтобы текст занял не больше max_lines строк."""
        full = f"{opener}{text}{closer}"
        if self.m.n_lines(full, font, pt, width) <= max_lines:
            return full
        words = text.split(" ")

        def cand(n: int) -> str:
            return opener + " ".join(words[:n]).rstrip(" ,;:—–-") + "…" + closer

        lo, hi = 1, len(words)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.m.n_lines(cand(mid), font, pt, width) <= max_lines:
                lo = mid
            else:
                hi = mid - 1
        return cand(lo)

    @staticmethod
    def _quote_body(q: str) -> str:
        q = q.strip()
        return q[1:-1] if q.startswith("«") and q.endswith("»") else q

    def _quote_lines(self, q: str) -> int:
        return self.m.n_lines(q, QUOTE_FONT, QUOTE_PT, TITLE_W)

    def _fit_quotes(self, ranked: list[str], max_lines: int) -> tuple[list[str], int]:
        if not ranked or max_lines <= 0:
            return [], 0
        first = self._clip(self._quote_body(ranked[0]), QUOTE_FONT, QUOTE_PT, TITLE_W, max_lines, "«", "»")
        out, used = [first], self._quote_lines(first)
        for extra in ranked[1:self.cfg.quotes.max_quotes]:
            q = f"«{self._quote_body(extra)}»"
            n = self._quote_lines(q)
            if used + n <= max_lines:
                out.append(q)
                used += n
        gaps = (len(out) - 1) * QUOTE_PARA_GAP_PT * 12700
        return out, used * QUOTE_LH + gaps

    def _base(self, t: Topic) -> _RowBase:
        title = self._clip(t.title, TITLE_FONT, TITLE_PT, TITLE_W, TITLE_MAX_LINES)
        title_h = self.m.n_lines(title, TITLE_FONT, TITLE_PT, TITLE_W) * TITLE_LH
        status, status_h = "", 0
        if t.status:
            w = STATUS_TXT_ICON_W if t.status_icon else STATUS_W - 2 * STATUS_INS_X
            status = self._clip(t.status, STATUS_FONT, STATUS_PT, w, STATUS_MAX_LINES)
            status_h = self.m.n_lines(status, STATUS_FONT, STATUS_PT, w) * QUOTE_LH + 2 * STATUS_INS_Y + STATUS_PAD
        return _RowBase(title, title_h, status, status_h)

    def _row(self, idx: int, t: Topic, base: _RowBase, max_lines: int) -> RowPlan:
        quotes, quote_h = self._fit_quotes(t.quotes, max_lines)
        text_h = base.title_h + (TITLE_GAP + quote_h if quotes else 0)
        box_h = max(STATUS_MIN_H, text_h, base.status_h)
        return RowPlan(idx, t, 0, base.title, base.title_h, quotes, quote_h, base.status, box_h)

    def measure_row(self, idx: int, t: Topic, max_lines: int | None = None, base: _RowBase | None = None) -> RowPlan:
        base = base or self._base(t)
        max_lines = max_lines or self.cfg.quotes.max_quote_lines
        row = self._row(idx, t, base, max_lines)
        while row.height > self.max_row_h and max_lines > 1:  # гарантия: строка влезает на любой слайд
            max_lines -= 1
            row = self._row(idx, t, base, max_lines)
        return row

    # ---- пагинация

    def plan(self, digest: Digest) -> list[SlidePlan]:
        """Перебирает общий лимит строк цитаты (от заданного вниз, но не меньше
        MIN_GLOBAL_QUOTE_LINES) и берёт план с наименьшим числом слайдов; при равенстве —
        с более полными цитатами. Убирает «висячие» слайды с одной строкой."""
        top = self.cfg.quotes.max_quote_lines
        best: list[SlidePlan] | None = None
        for k in range(top, min(top, MIN_GLOBAL_QUOTE_LINES) - 1, -1):
            cand = self._plan(digest, k)
            if best is None or len(cand) < len(best):
                best = cand
        return best

    def _plan(self, digest: Digest, quote_lines: int) -> list[SlidePlan]:
        slides = [SlidePlan(first=True)]
        bases = self._bases
        y = TOP_FIRST
        idx = 0
        for product, topics in digest.products:
            section: SectionPlan | None = None
            cursor = 0
            for t in topics:
                idx += 1
                base = bases.get(id(t)) or bases.setdefault(id(t), self._base(t))
                row = self.measure_row(idx, t, max_lines=quote_lines, base=base)
                top = cursor if section else y + ROW_OFFSET
                if top + row.height + CARD_PAD > BOTTOM:
                    # прежде чем переносить строку, ужимаем цитату, чтобы добить текущий слайд
                    for k in range(quote_lines - 1, MIN_SQUEEZE_LINES - 1, -1):
                        squeezed = self._row(idx, t, base, k)
                        if top + squeezed.height + CARD_PAD <= BOTTOM:
                            row = squeezed
                            break
                if top + row.height + CARD_PAD > BOTTOM:
                    if section:
                        self._close(section)
                    slides.append(SlidePlan(first=False))
                    y = TOP_OTHER
                    section = None
                    top = y + ROW_OFFSET
                if section is None:
                    section = SectionPlan(product=product, top=y)
                    slides[-1].sections.append(section)
                row.top = top
                section.rows.append(row)
                cursor = top + row.height + ROW_GAP
            if section:
                self._close(section)
                y = section.bottom + CARD_GAP
        return slides

    @staticmethod
    def _close(section: SectionPlan) -> None:
        last = section.rows[-1]
        section.bottom = last.top + last.height + CARD_PAD


# ----------------------------------------------------------------------------- отрисовка


class Renderer:
    def __init__(self, cfg: Config, measurer: Measurer):
        self.cfg = cfg
        self.m = measurer
        self.prs = Presentation(cfg.template_path)
        if not len(self.prs.slides):
            raise ValueError("В шаблоне нет эталонного слайда")
        self.ref = self.prs.slides[0]
        self.layout = self.ref.slide_layout
        tree = self.ref.shapes._spTree
        self.proto: dict[int, object] = {}
        def flat(v):
            if isinstance(v, tuple):
                for x in v:
                    yield from flat(x)
            else:
                yield v

        ids = {i for v in T.values() for i in flat(v)}
        for i in ids:
            try:
                self.proto[i] = find_by_id(tree, i)
            except KeyError:
                raise ValueError(f"В эталонном слайде шаблона нет фигуры id={i}. "
                                 "Шаблон изменён — обнови словарь T в render.py") from None

        self.title_rpr = run_styles(self.proto[T["row_title"]])[0]
        self.quote_rpr = run_styles(self.proto[T["quote"]])[0]
        self.status_rpr = run_styles(self.proto[T["status_text"]])[0]
        ensure_text_color(self.status_rpr, STATUS_TEXT_COLOR)  # у карточки fontRef=lt1 (белый)

    @staticmethod
    def _widen_kpi_label(card, label) -> None:
        """Подпись KPI («Сигналов / проанализировано»): в шаблоне у бокса внутренние отступы
        по 0.1", и длинное слово переносится в просмотрщиках без SB Sans. Убираем отступы
        (сдвигая бокс, чтобы текст остался на месте) и тянем бокс до правого края карточки."""
        cx, _, cw, _ = get_box(card)
        lx, ly, lw, lh = get_box(label)
        text_x = lx + LABEL_INS
        width = max(lw - 2 * LABEL_INS, cx + cw - text_x - KPI_LABEL_RIGHT_GAP)
        set_box(label, text_x, ly, width, lh)
        set_body_attr(label, lIns=0, rIns=0)

    def _clip_product(self, product: str) -> str:
        """Название продукта — строго в одну строку полосы шапки."""
        font, pt, w = "SB Sans Display Semibold", 11, PRODUCT_BOX_W - int(0.3 * IN)
        if self.m.width_emu(product, font, pt) <= w:
            return product
        lo, hi = 1, len(product)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.m.width_emu(product[:mid].rstrip() + "…", font, pt) <= w:
                lo = mid
            else:
                hi = mid - 1
        return product[:lo].rstrip() + "…"

    def c(self, key: str | int):
        el = clone(self.proto[T[key] if isinstance(key, str) else key])
        harden_nowrap(el)  # запас ширины однострочным подписям (см. ooxml.harden_nowrap)
        return el

    # ---- статичные блоки

    def _chrome(self, b: SlideBuilder, digest: Digest, first: bool) -> None:
        meta = digest.meta
        b.add(self.c("title"))
        date = self.c("date")
        lines = [meta.release_date, f"Период: {meta.period}" if meta.period else "", f"№ {meta.number}" if meta.number else ""]
        set_text(date, "\n".join(x for x in lines if x))
        dx, dy, dw, dh = get_box(date)
        set_box(date, dx + dw - DATE_BOX_W, dy, DATE_BOX_W, dh)  # algn=r: правый край на месте
        b.add(date)
        b.add(self.c("slide_num"))
        if first:
            values = {T["kpi_signals"]: meta.signals_total, T["kpi_sources"]: meta.sources_total,
                      T["kpi_active"]: meta.active_topics, T["kpi_new"]: meta.new_topics}
            for sid in T["top_block"]:
                el = self.c(sid)
                for vid, val in values.items():
                    try:
                        num_box = find_by_id(el, vid)
                        set_text(num_box, fmt_int(val))
                        set_box(num_box, cx=KPI_NUM_BOX_W)  # algn=l: левый край на месте
                    except KeyError:
                        pass
                for card_id, label_id in T["kpi_card_label"]:
                    try:
                        self._widen_kpi_label(find_by_id(el, card_id), find_by_id(el, label_id))
                    except KeyError:
                        pass
                b.add(el)
        for sid in T["legend"]:
            b.add(self.c(sid))

    # ---- строки

    def _count(self, t: Topic, top: int):
        el = self.c("count")
        s = run_styles(el)
        write_paragraphs(el, [[(fmt_int(t.cur), s[0]), ("\u00a0", s[1]), ("упом.", s[2])]])
        set_body_attr(el, wrap="square")
        return set_box(el, COL_CX - COUNT_BOX_W // 2, top, COUNT_BOX_W, COUNT_H)

    def _badge(self, dyn: int, top: int):
        el = self.c("badge")
        s = run_styles(el)
        up = dyn > 0
        arrow, value = ("▲" if up else "▼"), f"{dyn:+d}%"
        arrow_rpr, space_rpr, val_rpr = s[0], s[1], s[2]
        if up:
            recolor(arrow_rpr, RED_ARROW)
            for r in (space_rpr, val_rpr):
                recolor(r, RED_TEXT)
            set_fill(el, RED_FILL, alpha=60000)
            set_line(el, RED_TEXT)
        write_paragraphs(el, [[(arrow, arrow_rpr), (" ", space_rpr), (value, val_rpr)]])
        w = int(1.12 * (2 * BADGE_PAD + self.m.width_emu(arrow + " ", "SB Sans Display", 12)
                       + self.m.width_emu(value, "SB Sans Display Semibold", 14)))  # запас: пилюля не режет текст
        return set_box(el, COL_CX - w // 2, top + BADGE_DY, w, BADGE_H)

    def _row(self, b: SlideBuilder, r: RowPlan) -> None:
        t, top = r.topic, r.top

        num = set_box(self.c("num"), NUM_X, top)
        set_text(num, f"{r.index:02d}")
        b.add(num)

        # Название и цитаты — ОДИН текстбокс: переносы и отступы считает сам PowerPoint,
        # поэтому цитата физически не может наехать на название при другом переносе строк.
        text_box = set_box(self.c("row_title"), TITLE_X, top, TITLE_W, r.text_h)
        paragraphs = [[(r.title, self.title_rpr)]] + [[(q, self.quote_rpr)] for q in r.quotes]
        spacing = [None] + [TITLE_GAP / 12700] + [QUOTE_PARA_GAP_PT] * (len(r.quotes) - 1)
        write_paragraphs(text_box, paragraphs, spacing_before_pt=spacing[:len(paragraphs)])
        b.add(text_box)

        b.add(self._count(t, top))
        if t.resumed:
            b.add(set_box(self.c("icon_resumed"), COL_CX - ICON // 2, top + ICON_DY, ICON, ICON))
        elif t.is_new:
            b.add(set_box(self.c("icon_new"), COL_CX - ICON // 2, top + ICON_DY, ICON, ICON))
        elif t.dyn_pct:  # 0% — стабильность, по ТЗ ничего не показываем
            b.add(self._badge(t.dyn_pct, top))

        # Статус пишется прямо в карточку (anchor=ctr): центрирование делает PowerPoint,
        # место под иконку справа резервируется правым отступом.
        box_h = r.box_h
        card = set_box(self.c("status_box"), STATUS_X, top, STATUS_W, box_h)
        if r.status:
            if t.status_icon:
                l_ins = STATUS_TXT_ICON_X - STATUS_X
                r_ins = STATUS_W - l_ins - STATUS_TXT_ICON_W
            else:
                l_ins = r_ins = STATUS_INS_X
            set_body_attr(card, wrap="square", anchor="ctr", lIns=l_ins, rIns=r_ins,
                          tIns=STATUS_INS_Y, bIns=STATUS_INS_Y)
            write_paragraphs(card, [[(r.status, self.status_rpr)]])
        b.add(card)
        if t.status_icon:
            icon = self.c("icon_pain" if t.status_icon == "pain" else "icon_analysis")
            ix = STATUS_ICON_X if r.status else STATUS_X + (STATUS_W - ICON) // 2
            b.add(set_box(icon, ix, top + (box_h - ICON) // 2, ICON, ICON))

    def _section(self, b: SlideBuilder, s: SectionPlan) -> None:
        b.add(set_box(self.c("card"), CONTENT_X, s.top, CONTENT_W, s.bottom - s.top))
        header = set_box(self.c("header"), CONTENT_X, s.top)
        prod = find_by_id(header, T["header_product"])
        set_text(prod, self._clip_product(s.product))
        set_body_attr(prod, wrap="square")
        # у группы шапки масштаб по X = 1, поэтому EMU дочерних координат == EMU слайда
        set_box(prod, cx=PRODUCT_BOX_W)
        b.add(header)
        for r in s.rows:
            self._row(b, r)

    # ---- сборка

    def render(self, digest: Digest, plans: list[SlidePlan], out: str | Path) -> Path:
        src_part = self.ref.part
        originals = list(self.prs.slides._sldIdLst)
        for plan in plans:
            slide = self.prs.slides.add_slide(self.layout)
            b = SlideBuilder(slide, src_part)
            b.clear()
            # порядок = z-order: хром, затем карточки поверх фона, легенда в конце
            self._chrome(b, digest, plan.first)
            for s in plan.sections:
                self._section(b, s)
        # удаляем эталонные слайды; их уникальные части не попадут в файл
        lst = self.prs.slides._sldIdLst
        for sld in originals:
            rid = sld.rId
            lst.remove(sld)
            self.prs.part.drop_rel(rid)
        self._finalize(digest)
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.prs.save(out)
        return out

    def _finalize(self, digest: Digest) -> None:
        """Чистка пакета: неиспользуемые лейауты (с их картинками), чужая миниатюра, метаданные."""
        used = {s.slide_layout.part for s in self.prs.slides}
        for master in self.prs.slide_masters:
            for layout in list(master.slide_layouts):
                if layout.part not in used:
                    master.slide_layouts.remove(layout)  # фон 1672×941 ×3 — ~3 МБ мёртвого веса
        pkg_rels = self.prs.part.package._rels
        for rid, rel in list(pkg_rels.items()):
            if rel.reltype.endswith("/thumbnail"):  # превью исходного примера со старыми данными
                pkg_rels.pop(rid)
        meta = digest.meta
        now = datetime.now().replace(microsecond=0)
        cp = self.prs.core_properties
        cp.title = "Голос IT: дайджест" + (f" № {meta.number}" if meta.number else "") + f" от {meta.release_date}"
        cp.subject = f"Период: {meta.period}" if meta.period else ""
        cp.author = cp.last_modified_by = self.cfg.author
        cp.created = cp.modified = now
        cp.revision = 1
        cp.keywords = cp.comments = cp.category = ""
        self._reset_app_props()

    def _reset_app_props(self) -> None:
        """docProps/app.xml из шаблона: время правки, счётчики и оглавление чужого файла.
        Оставляем минимальный валидный набор; PowerPoint дозаполнит его при сохранении."""
        for part in self.prs.part.package.iter_parts():
            if str(part.partname) == "/docProps/app.xml":
                part._blob = (
                    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                    '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
                    'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
                    '<Application>Microsoft Office PowerPoint</Application>'
                    f'<Slides>{len(self.prs.slides)}</Slides><Notes>0</Notes><HiddenSlides>0</HiddenSlides>'
                    '<ScaleCrop>false</ScaleCrop><LinksUpToDate>false</LinksUpToDate>'
                    '<SharedDoc>false</SharedDoc><HyperlinksChanged>false</HyperlinksChanged>'
                    '<AppVersion>16.0000</AppVersion></Properties>'
                ).encode("utf-8")
                break
