"""Измерение и перенос текста по реальным метрикам шрифта (нужно для пагинации)."""
from __future__ import annotations

import glob
import logging
import os
import shutil
import subprocess
from functools import lru_cache

from PIL import ImageFont

from .config import FontConfig

log = logging.getLogger(__name__)

EMU_PER_INCH = 914400
EMU_PER_PT = 12700
_REF_SIZE = 200  # размер, на котором меряем, затем линейно масштабируем

# family в pptx -> (ключ в FontConfig, шаблоны имён файлов, запрос fc-match, фолбэки)
_FAMILIES = {
    "SB Sans Display": ("regular", ["*SBSansDisplay-Regular*", "*SB*Sans*Display*Regular*"],
                        "SB Sans Display:style=Regular", ["Inter:style=Regular", "DejaVu Sans"]),
    "SB Sans Display Semibold": ("semibold", ["*SBSansDisplay-SemiBold*", "*SB*Sans*Display*Semi*"],
                                 "SB Sans Display:style=SemiBold", ["Inter:style=SemiBold", "DejaVu Sans:style=Bold"]),
    "SB Sans Display Light": ("light", ["*SBSansDisplay-Light*", "*SB*Sans*Display*Light*"],
                              "SB Sans Display:style=Light", ["Inter:style=Light", "DejaVu Sans"]),
}


def _fc_match(query: str) -> tuple[str | None, str | None]:
    if not shutil.which("fc-match"):
        return None, None
    try:
        out = subprocess.run(["fc-match", "-f", "%{file}\t%{family}", query],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    if "\t" not in out:
        return None, None
    path, fam = out.split("\t", 1)
    return (path or None), fam


class Measurer:
    def __init__(self, cfg: FontConfig):
        self.cfg = cfg
        self._paths: dict[str, str] = {}
        exact = True
        for family, (attr, patterns, query, fallbacks) in _FAMILIES.items():
            path = getattr(cfg, attr)
            if not path and cfg.fonts_dir:
                for pat in patterns:
                    hits = glob.glob(os.path.join(cfg.fonts_dir, "**", pat), recursive=True)
                    hits = [h for h in hits if h.lower().endswith((".ttf", ".otf"))]
                    if hits:
                        path = sorted(hits)[0]
                        break
            if not path:
                p, fam = _fc_match(query)
                if p and fam and "sb sans" in fam.lower():
                    path = p
            if not path:
                exact = False
                for fb in fallbacks:
                    p, _ = _fc_match(fb)
                    if p:
                        path = p
                        break
            if not path:
                path = "DejaVuSans.ttf"
            self._paths[family] = path
        if not exact:
            log.warning("SB Sans Display не найден — переносы считаются по шрифту-заменителю (%s). "
                        "Укажи fonts.fonts_dir для точной вёрстки.", self._paths["SB Sans Display"])

    @lru_cache(maxsize=None)
    def _font(self, family: str) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(self._paths.get(family, self._paths["SB Sans Display"]), _REF_SIZE)

    @lru_cache(maxsize=65536)
    def width_emu(self, text: str, family: str, size_pt: float) -> int:
        px = self._font(family).getlength(text)
        return int(px / _REF_SIZE * size_pt * EMU_PER_PT * self.cfg.width_safety)

    @staticmethod
    def line_height_emu(size_pt: float, spacing: float = 1.2) -> int:
        return int(size_pt * spacing * EMU_PER_PT)

    def wrap(self, text: str, family: str, size_pt: float, width_emu: int) -> list[str]:
        """Жадный перенос по словам; длинные токены (URL, логи) режутся посимвольно."""
        lines: list[str] = []
        for para in text.split("\n"):
            words = para.split(" ")
            line = ""
            for word in words:
                cand = f"{line} {word}" if line else word
                if self.width_emu(cand, family, size_pt) <= width_emu:
                    line = cand
                    continue
                if line:
                    lines.append(line)
                    line = ""
                while self.width_emu(word, family, size_pt) > width_emu and len(word) > 1:
                    lo, hi = 1, len(word)
                    while lo < hi:  # максимальный префикс, который влезает
                        mid = (lo + hi + 1) // 2
                        if self.width_emu(word[:mid], family, size_pt) <= width_emu:
                            lo = mid
                        else:
                            hi = mid - 1
                    lines.append(word[:lo])
                    word = word[lo:]
                line = word
            lines.append(line)
        return lines

    def n_lines(self, text: str, family: str, size_pt: float, width_emu: int) -> int:
        return len(self.wrap(text, family, size_pt, width_emu))
