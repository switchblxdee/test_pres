"""Низкоуровневая работа с XML слайдов: клонирование, id, связи, текст, координаты."""
from __future__ import annotations

import copy
from lxml import etree
from pptx.oxml.ns import qn

R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NSMAP = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": R_NS,
}


def xp(el, path: str):
    return etree._Element.xpath(el, path, namespaces=NSMAP)


# ----------------------------------------------------------------------------- клонирование


def find_by_id(root, shape_id: int):
    """Фигура (sp/pic/grpSp/cxnSp) с данным cNvPr/@id внутри root (включая сам root)."""
    hits = xp(root, f"descendant-or-self::*[*/p:cNvPr[@id='{shape_id}']]")
    if not hits:
        raise KeyError(shape_id)
    return hits[0]


def strip_ext(el) -> None:
    """creationId-дубли PowerPoint терпит, но чище убрать."""
    for ext in xp(el, ".//p:cNvPr/a:extLst"):
        ext.getparent().remove(ext)


class SlideBuilder:
    """Собирает дерево фигур одного слайда: уникальные id + перенос связей картинок."""

    def __init__(self, slide, src_part):
        self.slide = slide
        self.tree = slide.shapes._spTree
        self.src_part = src_part
        self._rid_map: dict[str, str] = {}
        self._next_id = 2 + max((int(x) for x in xp(self.tree, ".//p:cNvPr/@id")), default=1)

    def clear(self) -> None:
        for child in list(self.tree):
            if child.tag in (qn("p:nvGrpSpPr"), qn("p:grpSpPr")):
                continue
            self.tree.remove(child)

    def _rebind(self, el) -> None:
        for node in el.iter():
            for attr, val in list(node.attrib.items()):
                if not attr.startswith("{" + R_NS + "}"):
                    continue
                if val not in self._rid_map:
                    rel = self.src_part.rels[val]
                    if rel.is_external:
                        new = self.slide.part.relate_to(rel.target_ref, rel.reltype, is_external=True)
                    else:
                        new = self.slide.part.relate_to(rel.target_part, rel.reltype)
                    self._rid_map[val] = new
                node.set(attr, self._rid_map[val])

    def _renumber(self, el) -> None:
        for c in el.iter(qn("p:cNvPr")):
            c.set("id", str(self._next_id))
            self._next_id += 1

    def add(self, el):
        """el — уже склонированный и отредактированный элемент."""
        self._renumber(el)
        self._rebind(el)
        self.tree.append(el)
        return el


def clone(el):
    c = copy.deepcopy(el)
    strip_ext(c)
    return c


# ----------------------------------------------------------------------------- геометрия


def _xfrm(el):
    if el.tag == qn("p:grpSp"):
        return xp(el, "./p:grpSpPr/a:xfrm")[0]
    return xp(el, "./p:spPr/a:xfrm")[0]


def get_box(el) -> tuple[int, int, int, int]:
    x = _xfrm(el)
    off, ext = x.find(qn("a:off")), x.find(qn("a:ext"))
    return int(off.get("x")), int(off.get("y")), int(ext.get("cx")), int(ext.get("cy"))


def set_box(el, x=None, y=None, cx=None, cy=None):
    """Для группы двигает/растягивает рамку группы (дочерние координаты пересчитает PowerPoint)."""
    xf = _xfrm(el)
    off, ext = xf.find(qn("a:off")), xf.find(qn("a:ext"))
    if x is not None:
        off.set("x", str(int(x)))
    if y is not None:
        off.set("y", str(int(y)))
    if cx is not None:
        ext.set("cx", str(int(cx)))
    if cy is not None:
        ext.set("cy", str(int(cy)))
    return el


# ----------------------------------------------------------------------------- текст


def run_styles(el) -> list:
    """Копии rPr всех ранов фигуры по порядку — палитра стилей для перезаписи."""
    out = []
    for r in xp(el, ".//p:txBody/a:p/a:r"):
        rpr = r.find(qn("a:rPr"))
        out.append(copy.deepcopy(rpr) if rpr is not None else None)
    return out


def _make_run(text: str, rpr):
    r = etree.Element(qn("a:r"))
    if rpr is not None:
        rp = copy.deepcopy(rpr)
        for a in ("err", "smtClean", "dirty"):
            rp.attrib.pop(a, None)
        r.append(rp)
    t = etree.SubElement(r, qn("a:t"))
    t.text = text
    return r


def _set_spacing_before(pp, pt: float) -> None:
    for old in pp.findall(qn("a:spcBef")):
        pp.remove(old)
    sb = etree.Element(qn("a:spcBef"))
    etree.SubElement(sb, qn("a:spcPts")).set("val", str(int(round(pt * 100))))
    # порядок по схеме: lnSpc, spcBef, spcAft, bu*, tabLst, defRPr
    ln = pp.find(qn("a:lnSpc"))
    pp.insert(list(pp).index(ln) + 1 if ln is not None else 0, sb)


def write_paragraphs(el, paragraphs, spacing_before_pt=None):
    """Перезаписывает текст фигуры.

    paragraphs: список абзацев; абзац — список сегментов (text, rPr). Символ '\\n' внутри
    text превращается в <a:br/> с тем же стилем. pPr и endParaRPr первого абзаца
    копируются во все абзацы.
    spacing_before_pt: число (для всех абзацев, кроме первого) или список по абзацам.
    """
    paragraphs = list(paragraphs)
    if spacing_before_pt is None or isinstance(spacing_before_pt, (int, float)):
        spacing = [None] + [spacing_before_pt] * (len(paragraphs) - 1)
    else:
        spacing = list(spacing_before_pt)
    body = xp(el, "./p:txBody")[0]
    old_ps = body.findall(qn("a:p"))
    proto = old_ps[0]
    ppr = proto.find(qn("a:pPr"))
    end = proto.find(qn("a:endParaRPr"))
    for p in old_ps:
        body.remove(p)
    for i, segs in enumerate(paragraphs):
        p = etree.SubElement(body, qn("a:p"))
        pp = copy.deepcopy(ppr) if ppr is not None else None
        if spacing[i]:
            if pp is None:
                pp = etree.Element(qn("a:pPr"))
            _set_spacing_before(pp, spacing[i])
        if pp is not None:
            p.append(pp)
        for text, rpr in segs:
            for j, chunk in enumerate(text.split("\n")):
                if j:
                    br = etree.SubElement(p, qn("a:br"))
                    if rpr is not None:
                        br.append(copy.deepcopy(rpr))
                if chunk:
                    p.append(_make_run(chunk, rpr))
        last_rpr = segs[-1][1] if segs else None
        if last_rpr is not None:
            # endParaRPr = стиль последнего рана, иначе PowerPoint берёт высоту строки от шаблонного
            e = copy.deepcopy(last_rpr)
            e.tag = qn("a:endParaRPr")
            for a in ("err", "smtClean", "dirty"):
                e.attrib.pop(a, None)
            p.append(e)
        elif end is not None:
            p.append(copy.deepcopy(end))
    return el


def ensure_text_color(rpr, srgb: str = "000000") -> None:
    """Явный цвет текста — для фигур с fontRef lt1 (иначе текст будет белым)."""
    if rpr is None or rpr.find(qn("a:solidFill")) is not None:
        return
    fill = etree.Element(qn("a:solidFill"))
    etree.SubElement(fill, qn("a:srgbClr")).set("val", srgb)
    # по схеме CT_TextCharacterProperties заливка идёт сразу после a:ln
    after_fill = {qn(t) for t in ("a:effectLst", "a:effectDag", "a:highlight", "a:uLnTx", "a:uLn",
                                  "a:uFillTx", "a:uFill", "a:latin", "a:ea", "a:cs", "a:sym",
                                  "a:hlinkClick", "a:hlinkMouseOver", "a:rtl", "a:extLst")}
    anchor = next((c for c in rpr if c.tag in after_fill), None)
    if anchor is not None:
        anchor.addprevious(fill)
    else:
        rpr.append(fill)


def set_text(el, text: str, style_index: int = 0):
    styles = run_styles(el)
    rpr = styles[min(style_index, len(styles) - 1)] if styles else None
    return write_paragraphs(el, [[(text, rpr)]])


def set_body_attr(el, **attrs) -> None:
    bp = xp(el, "./p:txBody/a:bodyPr")[0]
    for k, v in attrs.items():
        bp.set(k, str(v))


def recolor(rpr_or_el, srgb: str) -> None:
    """Заменяет заливку (solidFill) на srgb — для красного бейджа роста."""
    for fill in xp(rpr_or_el, "descendant-or-self::a:solidFill"):
        for c in list(fill):
            fill.remove(c)
        etree.SubElement(fill, qn("a:srgbClr")).set("val", srgb)


def set_fill(el, srgb: str, alpha: int | None = None) -> None:
    fill = xp(el, "./p:spPr/a:solidFill")[0]
    for c in list(fill):
        fill.remove(c)
    clr = etree.SubElement(fill, qn("a:srgbClr"))
    clr.set("val", srgb)
    if alpha is not None:
        etree.SubElement(clr, qn("a:alpha")).set("val", str(alpha))


def set_line(el, srgb: str) -> None:
    for fill in xp(el, "./p:spPr/a:ln/a:solidFill"):
        for c in list(fill):
            fill.remove(c)
        etree.SubElement(fill, qn("a:srgbClr")).set("val", srgb)


def _is_transparent(sp) -> bool:
    """Текстбокс без заливки и обводки: его ширина невидима, расширять безопасно."""
    sppr = sp.find(qn("p:spPr"))
    if sppr is None:
        return False
    for tag in ("a:solidFill", "a:gradFill", "a:blipFill", "a:pattFill"):
        if sppr.find(qn(tag)) is not None:
            return False
    ln = sppr.find(qn("a:ln"))
    if ln is not None and ln.find(qn("a:noFill")) is None and len(ln):
        return False
    return sp.find(qn("p:style")) is None  # у фигур со стилем есть заливка из темы


def harden_nowrap(el, factor: float = 1.5, extra_emu: int = 228600) -> None:
    """Расширяет прозрачные однострочные (wrap=none) текстбоксы внутри el.

    Просмотрщики без SB Sans подставляют более широкий шрифт и часто игнорируют
    wrap="none" — текст в тесном боксе переносится и наезжает на соседей. Запас
    ширины это исключает; wrap переводится в square, чтобы выравнивание абзаца
    одинаково соблюдалось во всех редакторах.
    """
    for sp in el.iter(qn("p:sp")):
        body = sp.find(qn("p:txBody"))
        if body is None or not _is_transparent(sp):
            continue
        bp = body.find(qn("a:bodyPr"))
        if bp is None or bp.get("wrap") != "none":
            continue
        xf = xp(sp, "./p:spPr/a:xfrm")
        if not xf:
            continue
        off, ext = xf[0].find(qn("a:off")), xf[0].find(qn("a:ext"))
        x, w = int(off.get("x")), int(ext.get("cx"))
        new_w = int(w * factor + extra_emu)
        algn = next(iter(xp(body, "./a:p/a:pPr/@algn") or xp(body, "./a:lstStyle/a:lvl1pPr/@algn")), "l")
        if algn == "r":
            off.set("x", str(x + w - new_w))
        elif algn == "ctr":
            off.set("x", str(x - (new_w - w) // 2))
        ext.set("cx", str(new_w))
        # wrap=none в широком боксе редакторы трактуют по-разному (LibreOffice центрирует
        # текст независимо от algn) — переводим в square: выравнивание абзаца работает везде,
        # а запаса ширины хватает, чтобы строка не переносилась
        bp.set("wrap", "square")
