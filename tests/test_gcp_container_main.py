import pytest

from app.gcp import container_main


def test_container_main_unknown_role_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPOTBALLER_CONTAINER_ROLE", "not-a-role")
    with pytest.raises(SystemExit) as exc:
        container_main.main()
    assert exc.value.code == 1
