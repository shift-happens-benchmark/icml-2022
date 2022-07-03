"""Global configuration options for the benchmark."""

import dataclasses
import os


@dataclasses.dataclass(frozen=True)
class Config:
    """Global configuration for the benchmark.

    Config options can be edited by the following order:
    1. By setting variables explicitly when getting the instance via get_config()
    2. By setting an environment variable, prefixed as "SH_VARIABLE_NAME"
    3. By relying on the default values defined in this class.
    """

    __instance = None

    @classmethod
    def _reset_instance(cls):
        cls.__instance = None

    @classmethod
    def _init_instance(cls, **init_kwargs):
        prefix = "SH_"
        for field in dataclasses.fields(cls):
            if field.name in init_kwargs:
                continue
            environment_variable_key = prefix + field.name.upper()
            if environment_variable_key in os.environ:
                init_kwargs[field.name] = os.environ[environment_variable_key]
        return cls(**init_kwargs)

    @classmethod
    def get_instance(cls, **init_kwargs):
        if cls.__instance is None:
            cls.__instance = cls._init_instance(**init_kwargs)
        elif len(init_kwargs) > 0:
            if cls._init_instance(**init_kwargs) != cls.__instance:
                raise ValueError(
                    "Invalid configuration options specified. The global config "
                    f"was already initialized with values {cls.__instance}, but a "
                    "second initialization was request with incompatible arguments "
                    f"{init_kwargs}."
                )
        return cls.__instance

    imagenet_validation_path: str = "shifthappens/imagenet"
    """The imagenet validation path."""

    cache_directory_path: str = "shifthappens/cache"
    """The caching directory for model results (either absolute or relative to working directory).
    If the folder does not exist, it will be created."""

    verbose: bool = False
    """Show additional log messages on stderr (like progress bars)"""

    def __contains__(self, key):
        return key in self.__dict__


def get_config(**init_kwargs):
    return Config().get_instance(**init_kwargs)


def __getattr__(key):
    config = get_config()
    if key in config:
        return getattr(config, key)
    raise AttributeError(key)
