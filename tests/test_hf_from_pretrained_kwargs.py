import os

import pytest

from app.pipeline import pretrained_stack as ps


def test_hf_from_pretrained_kwargs_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_test_dummy")
    kw = ps.hf_from_pretrained_kwargs()
    assert kw.get("token") == "hf_test_dummy"


def test_hf_from_pretrained_kwargs_local_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("SPOTBALLER_HF_LOCAL_ONLY", "1")
    kw = ps.hf_from_pretrained_kwargs()
    assert kw.get("local_files_only") is True


def test_apply_spotballer_hf_offline_sets_hub_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setenv("SPOTBALLER_HF_OFFLINE", "1")
    # Re-run apply (module already imported once without this env in other tests)
    ps._apply_hf_hub_env_defaults()
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def test_ensure_hf_no_proxy_appends_hub_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("SPOTBALLER_HF_RESPECT_PROXY", raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9999")
    ps._ensure_hf_no_proxy_for_hub()
    assert "huggingface.co" in (os.environ.get("NO_PROXY") or "")


def test_hf_from_pretrained_kwargs_auto_local_when_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("SPOTBALLER_HF_LOCAL_ONLY", raising=False)
    monkeypatch.delenv("SPOTBALLER_HF_DISABLE_AUTO_OFFLINE", raising=False)

    monkeypatch.setattr(ps, "_hf_config_cached_locally", lambda rid: rid == "org/model")
    kw = ps.hf_from_pretrained_kwargs("org/model")
    assert kw.get("local_files_only") is True
