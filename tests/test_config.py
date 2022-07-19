import os

import pytest

import shifthappens.config


def reset_instance():
    """Reset the singleton instance at the start of each test."""
    shifthappens.config.Config._reset_instance()


def test_init_config():
    reset_instance()
    cfg = shifthappens.config.get_config()
    assert cfg is not None
    assert isinstance(cfg, shifthappens.config.Config)


def test_env_config():
    reset_instance()
    os.environ["SH_CACHE_DIRECTORY_PATH"] = "/tmp"
    cfg = shifthappens.config.get_config()
    assert cfg.cache_directory_path == "/tmp"


def test_default_args_config():
    reset_instance()
    os.environ["SH_CACHE_DIRECTORY_PATH"] = "/set_by_env"
    cfg = shifthappens.config.get_config(cache_directory_path="/set_by_arg")
    assert cfg.cache_directory_path == "/set_by_arg"


def test_default_args_wrong_override_config():
    reset_instance()
    os.environ["SH_CACHE_DIRECTORY_PATH"] = "/set_by_env"
    cfg = shifthappens.config.get_config()
    assert cfg.cache_directory_path == "/set_by_env"

    # get config again with correct arg is fine
    cfg = shifthappens.config.get_config(cache_directory_path="/set_by_env")
    assert cfg.cache_directory_path == "/set_by_env"

    # get config with different arg will throw an error
    with pytest.raises(ValueError):
        cfg = shifthappens.config.get_config(cache_directory_path="/set_by_arg")


def test_retrieve_directly():
    reset_instance()
    cfg = shifthappens.config.get_config()
    shifthappens.config.cache_directory_path == cfg.cache_directory_path
