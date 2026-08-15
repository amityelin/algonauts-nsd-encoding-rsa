from pathlib import Path

import pytest

from algonauts_rsa import config


def test_explicit_paths_take_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGONAUTS_DATA_ROOT", str(tmp_path / "env-data"))
    monkeypatch.setenv("ALGONAUTS_OUTPUT_ROOT", str(tmp_path / "env-output"))
    paths = config.get_paths(tmp_path / "explicit-data", tmp_path / "explicit-output")
    assert paths.data_root == tmp_path / "explicit-data"
    assert paths.output_root == tmp_path / "explicit-output"


def test_environment_paths_take_precedence_over_colab(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGONAUTS_DATA_ROOT", str(tmp_path / "env-data"))
    monkeypatch.setenv("ALGONAUTS_OUTPUT_ROOT", str(tmp_path / "env-output"))
    monkeypatch.setattr(config, "_running_in_colab", lambda: True)
    paths = config.get_paths()
    assert paths.data_root == tmp_path / "env-data"
    assert paths.output_root == tmp_path / "env-output"


def test_colab_defaults_only_inside_colab(monkeypatch):
    monkeypatch.delenv("ALGONAUTS_DATA_ROOT", raising=False)
    monkeypatch.delenv("ALGONAUTS_OUTPUT_ROOT", raising=False)
    monkeypatch.setattr(config, "_running_in_colab", lambda: True)
    paths = config.get_paths()
    assert paths.data_root.as_posix().startswith("/content/drive/")
    assert paths.output_root.as_posix() == "/content/drive/MyDrive/algonauts_outputs"


def test_unconfigured_local_paths_raise_actionable_error(monkeypatch):
    monkeypatch.delenv("ALGONAUTS_DATA_ROOT", raising=False)
    monkeypatch.delenv("ALGONAUTS_OUTPUT_ROOT", raising=False)
    monkeypatch.setattr(config, "_running_in_colab", lambda: False)
    with pytest.raises(RuntimeError, match="ALGONAUTS_DATA_ROOT.*get_paths"):
        config.get_paths()


@pytest.mark.parametrize("mode", ["smoke", "fast", "full"])
def test_configuration_name_round_trip(mode):
    assert config.get_config(mode).name == mode


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError, match="Unknown run mode"):
        config.get_config("unknown")
