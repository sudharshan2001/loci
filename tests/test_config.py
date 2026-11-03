from loci.config import load_config, repo_path


def test_default_config_loads() -> None:
    cfg = load_config()
    assert isinstance(cfg["cloud"]["enabled"], bool)
    assert {"home", "street"} <= set(cfg["scenes"])


def test_no_cloud_env_wins(monkeypatch) -> None:
    from loci.agent.agent import answer
    from loci.cloud import meta_api

    monkeypatch.setenv("LOCI_CLOUD", "0")
    assert not meta_api.enabled()
    # with cloud off, the device answers: the Glimmer path is taken without any Meta call
    calls = []
    monkeypatch.setattr("loci.agent.agent.ask", lambda q, mem, url: calls.append(url) or {"answer": "ok"})
    answer("where?", None, lambda: "http://local")
    assert calls == ["http://local"]


def test_cloud_failure_falls_back_to_the_device(monkeypatch) -> None:
    from loci.agent.agent import answer
    from loci.cloud import meta_api

    monkeypatch.setenv("LOCI_CLOUD", "1")

    def fake_ask(q, mem, url):
        if url == meta_api.BASE_URL:
            raise RuntimeError("Meta API unreachable")
        return {"answer": "ok", "url": url}

    monkeypatch.setattr("loci.agent.agent.ask", fake_ask)
    assert answer("where?", None, lambda: "http://local")["url"] == "http://local"


def test_street_does_not_redact_documents() -> None:
    # shop signboards must stay readable on the street
    prompts = load_config()["scenes"]["street"]["redact_prompts"]
    assert not any("document" in p for p in prompts)


def test_repo_path_is_absolute() -> None:
    assert repo_path("configs").is_absolute()


def test_memory_db_env_override(monkeypatch) -> None:
    monkeypatch.setenv("LOCI_MEMORY_DB", "data/memory_poc.sqlite")
    load_config.cache_clear()
    try:
        assert load_config()["paths"]["memory_db"] == "data/memory_poc.sqlite"
    finally:
        monkeypatch.delenv("LOCI_MEMORY_DB")
        load_config.cache_clear()
