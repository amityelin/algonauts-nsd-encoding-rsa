import pytest

from algonauts_rsa import config


def test_explicit_paths_take_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGONAUTS_DATA_ROOT", str(tmp_path / "env-data"))
    monkeypatch.setenv("ALGONAUTS_OUTPUT_ROOT", str(tmp_path / "env-output"))
    paths = config.get_paths(tmp_path / "explicit-data", tmp_path / "explicit-output")
    assert paths.data_root == tmp_path / "explicit-data"
    assert paths.output_root == tmp_path / "explicit-output"


def test_environment_paths_are_used(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGONAUTS_DATA_ROOT", str(tmp_path / "env-data"))
    monkeypatch.setenv("ALGONAUTS_OUTPUT_ROOT", str(tmp_path / "env-output"))
    paths = config.get_paths()
    assert paths.data_root == tmp_path / "env-data"
    assert paths.output_root == tmp_path / "env-output"


def test_unconfigured_paths_raise_actionable_error(monkeypatch):
    monkeypatch.delenv("ALGONAUTS_DATA_ROOT", raising=False)
    monkeypatch.delenv("ALGONAUTS_OUTPUT_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="ALGONAUTS_DATA_ROOT.*get_paths"):
        config.get_paths()


def test_user_supplied_colab_paths_are_preserved():
    paths = config.get_paths(
        "/content/drive/MyDrive/algonauts_data",
        "/content/drive/MyDrive/algonauts_results",
    )
    assert paths.data_root.as_posix() == "/content/drive/MyDrive/algonauts_data"
    assert paths.output_root.as_posix() == "/content/drive/MyDrive/algonauts_results"


@pytest.mark.parametrize("mode", ["smoke", "fast", "full"])
def test_configuration_name_round_trip(mode):
    assert config.get_config(mode).name == mode


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError, match="Unknown run mode"):
        config.get_config("unknown")
