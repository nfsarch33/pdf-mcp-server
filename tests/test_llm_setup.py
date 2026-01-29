from types import SimpleNamespace

from pdf_mcp import llm_setup


def test_parse_ollama_list_output_returns_models():
    output = """NAME            ID              SIZE      MODIFIED
qwen2.5:7b     abcdef123456    4.1 GB   2 days ago
qwen2.5:1.5b   9876543210ab    1.1 GB   5 days ago
"""
    models = llm_setup.parse_ollama_list_output(output)
    assert models == {"qwen2.5:7b", "qwen2.5:1.5b"}


def test_parse_ollama_list_output_ignores_blanks():
    output = """
NAME            ID              SIZE      MODIFIED

qwen2.5:7b     abcdef123456    4.1 GB   2 days ago
"""
    models = llm_setup.parse_ollama_list_output(output)
    assert models == {"qwen2.5:7b"}


def test_get_ollama_model_name_default(monkeypatch):
    monkeypatch.delenv(llm_setup.OLLAMA_MODEL_ENV, raising=False)
    assert llm_setup.get_ollama_model_name() == llm_setup.DEFAULT_OLLAMA_MODEL


def test_get_ollama_model_name_env_override(monkeypatch):
    monkeypatch.setenv(llm_setup.OLLAMA_MODEL_ENV, "custom-model:latest")
    assert llm_setup.get_ollama_model_name() == "custom-model:latest"


def test_ollama_list_models_returns_empty_when_not_installed(monkeypatch):
    monkeypatch.setattr(llm_setup, "ollama_is_installed", lambda: False)
    models = llm_setup.ollama_list_models()
    assert models == set()


def test_ollama_list_models_returns_empty_on_failure(monkeypatch):
    monkeypatch.setattr(llm_setup, "ollama_is_installed", lambda: True)

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="error")

    models = llm_setup.ollama_list_models(run=fake_run)
    assert models == set()


def test_ollama_model_installed_detects_model(monkeypatch):
    monkeypatch.setattr(llm_setup, "ollama_is_installed", lambda: True)

    def fake_run(*_args, **_kwargs):
        output = "NAME ID SIZE MODIFIED\nqwen2.5:7b abc 1GB 2d\n"
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    assert llm_setup.ollama_model_installed("qwen2.5:7b", run=fake_run) is True
    assert llm_setup.ollama_model_installed("missing", run=fake_run) is False


# ============================================================================
# Tests for local server diagnostics (v0.9.5+)
# ============================================================================


def test_get_local_server_health_returns_none_when_unavailable(monkeypatch):
    """get_local_server_health returns None if server not running."""
    # Patch requests to raise connection error
    class FakeRequests:
        @staticmethod
        def get(*_args, **_kwargs):
            raise ConnectionError("server not running")

    monkeypatch.setattr(llm_setup, "requests", FakeRequests, raising=False)
    result = llm_setup.get_local_server_health()
    assert result is None


def test_get_local_server_models_returns_none_when_unavailable(monkeypatch):
    """get_local_server_models returns None if server not running."""
    class FakeRequests:
        @staticmethod
        def get(*_args, **_kwargs):
            raise ConnectionError("server not running")

    monkeypatch.setattr(llm_setup, "requests", FakeRequests, raising=False)
    result = llm_setup.get_local_server_models()
    assert result is None


def test_local_model_server_url_default():
    """LOCAL_MODEL_SERVER_URL should default to localhost:8100."""
    assert "localhost:8100" in llm_setup.LOCAL_MODEL_SERVER_URL


def test_get_local_server_health_exists():
    """get_local_server_health function should exist."""
    assert hasattr(llm_setup, "get_local_server_health")
    assert callable(llm_setup.get_local_server_health)


def test_get_local_server_models_exists():
    """get_local_server_models function should exist."""
    assert hasattr(llm_setup, "get_local_server_models")
    assert callable(llm_setup.get_local_server_models)


def test_local_vlm_model_default():
    """LOCAL_VLM_MODEL should default to qwen3-vl-30b-a3b."""
    assert hasattr(llm_setup, "LOCAL_VLM_MODEL")
    assert "qwen" in llm_setup.LOCAL_VLM_MODEL.lower()
