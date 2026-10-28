import pytest

from golos_digest import DigestAgent, DigestResult
from tests.conftest import markup_row, summary_row
from tests.test_quotes import FakeModel


def _xlsx(make_xlsx):
    s = [summary_row("P", "Тема A", 10, 5), summary_row("P", "Тема B", 3, 0)]
    m = [markup_row("P", "Тема A", "Сервис падает с ошибкой 500 при загрузке, повтор не помогает."),
         markup_row("P", "Тема A", "Не могу подключиться: таймаут и 401, раньше всё работало."),
         markup_row("P", "Тема B", "Нет доступа к инструменту, роль аналитик.")]
    return make_xlsx(s, m)


def test_agent_class_run_and_reuse(make_xlsx, cfg, tmp_path):
    cfg.llm.backend = "gigachat"
    model = FakeModel(mode="tool")
    agent = DigestAgent(cfg, llm=model)
    assert agent.llm is model
    path = _xlsx(make_xlsx)

    r1 = agent.run(path, tmp_path / "a.pptx")
    assert isinstance(r1, DigestResult) and r1.path.exists()
    assert (r1.n_topics, r1.llm_calls, r1.llm_failures) == (2, 1, 0)  # у темы B один кандидат — без LLM

    r2 = agent(path, tmp_path / "b.pptx")  # тот же экземпляр, второй запуск — всё из кэша
    assert r2.llm_calls == 0 and model.calls == 1


def test_agent_default_output_in_cwd_absolute(make_xlsx, cfg, tmp_path, monkeypatch):
    path = _xlsx(make_xlsx)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    r = DigestAgent(cfg).run(path)
    assert r.path.is_absolute() and r.path.parent == work and r.path.name.startswith("golos_it_digest_")
    assert r.path.exists()


def test_agent_failures_are_reported(make_xlsx, cfg, tmp_path):
    cfg.llm.backend, cfg.llm.retries = "gigachat", 1
    r = DigestAgent(cfg, llm=FakeModel(mode="fail")).run(_xlsx(make_xlsx), tmp_path / "f.pptx")
    assert r.path.exists() and r.llm_failures == r.llm_calls == 1


def test_agent_from_yaml_and_missing_input(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text("llm:\n  backend: heuristic\n", encoding="utf-8")
    agent = DigestAgent.from_yaml(y)
    assert agent.llm is None and "select_quotes" in agent.draw()
    with pytest.raises(FileNotFoundError):
        agent.run(tmp_path / "nope.xlsx")
