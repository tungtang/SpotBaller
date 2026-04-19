"""gcloud config path fallback for VM mode (sandbox / non-writable ~/.config/gcloud)."""

from __future__ import annotations

import os

import pytest

from app.gcp import vm_runner


@pytest.fixture(autouse=True)
def _reset_gcloud_cache() -> None:
    vm_runner.reset_gcloud_config_cache_for_tests()
    yield
    vm_runner.reset_gcloud_config_cache_for_tests()


def test_effective_none_when_home_config_writable(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLOUDSDK_CONFIG", raising=False)
    monkeypatch.delenv("SPOTBALLER_GCLOUD_CONFIG", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    home_gc = tmp_path / ".config" / "gcloud"
    home_gc.mkdir(parents=True)
    assert vm_runner._effective_gcloud_config_dir() is None


def test_explicit_spotballer_gcloud_config(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    dest = tmp_path / "custom_gcloud"
    monkeypatch.setenv("SPOTBALLER_GCLOUD_CONFIG", str(dest))
    got = vm_runner._effective_gcloud_config_dir()
    assert got == dest.resolve()
    assert got.is_dir()
    assert (got / "logs").is_dir()


@pytest.mark.skipif(os.name == "nt", reason="chmod semantics differ on Windows")
def test_fallback_runtime_dir_when_home_not_writable(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLOUDSDK_CONFIG", raising=False)
    monkeypatch.delenv("SPOTBALLER_GCLOUD_CONFIG", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    home_gc = tmp_path / ".config" / "gcloud"
    home_gc.mkdir(parents=True)
    (home_gc / "credentials.db").write_text("seed-cred")
    home_gc.chmod(0o555)
    try:
        got = vm_runner._effective_gcloud_config_dir()
        assert got is not None
        assert got.name == "gcloud_config"
        assert (got / "credentials.db").read_text() == "seed-cred"
    finally:
        home_gc.chmod(0o755)


def test_gcloud_env_sets_cloudsdk_when_fallback(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLOUDSDK_CONFIG", raising=False)
    monkeypatch.delenv("SPOTBALLER_GCLOUD_CONFIG", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    home_gc = tmp_path / ".config" / "gcloud"
    home_gc.mkdir(parents=True)
    (home_gc / "credentials.db").write_text("x")
    if os.name != "nt":
        home_gc.chmod(0o555)
        try:
            env = vm_runner._gcloud_env()
            assert "CLOUDSDK_CONFIG" in env
            assert env["CLOUDSDK_CONFIG"].endswith("runtime/gcloud_config")
        finally:
            home_gc.chmod(0o755)
    else:
        env = vm_runner._gcloud_env()
        assert isinstance(env, dict)
