"""CLI: python -m golos_digest input.xlsx -o digest.pptx [-c config.yaml]"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

from .config import load_config
from .graph import build


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="golos_digest", description="Генерация дайджеста «Голос IT» из xlsx")
    ap.add_argument("input", help="xlsx с листами «разметка» и «свод»")
    ap.add_argument("-o", "--output", default=f"golos_it_digest_{date.today():%Y%m%d}.pptx")
    ap.add_argument("-c", "--config", help="YAML с переопределениями (см. config.example.yaml)")
    ap.add_argument("--template", help="pptx-шаблон (по умолчанию template/template.pptx)")
    ap.add_argument("--backend", choices=["gigachat", "openai", "heuristic"], help="переопределить llm.backend")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "openai", "gigachat", "langchain_gigachat"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = load_config(args.config)
    if args.template:
        cfg.template_path = args.template
    if args.backend:
        cfg.llm.backend = args.backend
    try:
        build(args.input, args.output, cfg)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        logging.getLogger("golos_digest").error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
