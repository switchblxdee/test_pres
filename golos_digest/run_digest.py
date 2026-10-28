"""Пример запуска агента со своей моделью GigaChat (mTLS, ИФТ-контур).

    python run_digest.py выгрузка.xlsx [digest.pptx]
"""
import logging
import os
import sys

from langchain_gigachat.chat_models import GigaChat

from golos_digest import DigestAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
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
        timeout=99999,
        repetition_penalty=1.05,
        max_retries=0,  # ретраи делает цепочка агента (llm.retries), иначе они перемножаются
    )


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    # Остальные настройки (шрифты, лимиты цитат, параллелизм) — из config.yaml, если он есть.
    agent = DigestAgent("config.yaml" if os.path.exists("config.yaml") else None, llm=make_model())
    result = agent.run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(result)


if __name__ == "__main__":
    main()
