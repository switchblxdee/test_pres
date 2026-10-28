"""Запуск агента со своей моделью GigaChat (mTLS, ИФТ-контур).

    python run_digest.py                         # пути из INPUT_XLSX / OUTPUT_PPTX ниже (кнопка Run в VS Code)
    python run_digest.py выгрузка.xlsx [out.pptx] # или из аргументов
"""
import logging
import os
import sys
import traceback
from pathlib import Path

from langchain_gigachat.chat_models import GigaChat

from golos_digest import DigestAgent

# ---- что и куда (используется, если не переданы аргументы командной строки)
INPUT_XLSX = "выгрузка.xlsx"
OUTPUT_PPTX = None   # None -> ./golos_it_digest_ГГГГММДД.pptx в текущей папке

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S", force=True)
for noisy in ("httpx", "httpcore", "gigachat", "langchain_gigachat"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def make_model() -> GigaChat:
    return GigaChat(
        ca_bundle_file="/home/work/23101393@sigma.sbrf.ru/vs_code/serv.pem",
        cert_file="/home/work/23101393@sigma.sbrf.ru/vs_code/CI03121738-DEV-AIBOLIT.pem",
        key_file="/home/work/23101393@sigma.sbrf.ru/vs_code/CI03121738-DEV-AIBOLIT.key",
        base_url="https://gigachat-ift.sberdevices.delta.sbrf.ru/v1",
        model="Qwen3.5-397b",
        verify_ssl_certs=True,
        temperature=0.,
        timeout=600,      # было 99999: зависший запрос ждал бы ~28 ч и прогон выглядел бы замершим
        repetition_penalty=1.05,
        max_retries=0,    # ретраи делает цепочка агента (llm.retries), иначе они перемножаются
    )


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else INPUT_XLSX
    dst = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_PPTX
    if not Path(src).expanduser().is_file():
        print(f"\nНе найден входной файл: {Path(src).expanduser().resolve()}\n"
              f"Текущая папка: {Path.cwd()}\n"
              f"Укажи путь аргументом или поправь INPUT_XLSX в {Path(__file__).name}.")
        return 1
    try:
        # Остальные настройки (шрифты, лимиты цитат, параллелизм) — из config.yaml рядом, если он есть.
        cfg = "config.yaml" if os.path.exists("config.yaml") else None
        result = DigestAgent(cfg, llm=make_model()).run(src, dst)
    except Exception:
        print("\nГенерация упала. Пришли этот текст целиком:\n")
        traceback.print_exc()
        return 1
    print(f"\nПрезентация: {result.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
