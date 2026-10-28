from loci.config import load_config, repo_path


def test_default_config_loads() -> None:
    cfg = load_config()
    assert cfg["cloud"]["enabled"] is False
    assert {"home", "street"} <= set(cfg["scenes"])


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
