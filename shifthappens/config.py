"""Global configuration options for the benchmark."""

import dataclasses
import os
from typing import Any
from typing import List


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
        """
        Initializes config with provided arguments. If no arguments were provided,
        config would be initialized with corresponding environment variables if
        they exist. Otherwise, it will be initialized with default field values
        defined in :py:class:`Config <shifthappens.config.Config>`.
        Args:
            **init_kwargs: Values for initializing :py:class:`Config <shifthappens.config.Config>`
         fields.

        Returns:
            Initialized :py:class:`Config <shifthappens.config.Config>`.
        """
        if cls.__instance is None:
            cls.__instance = cls._init_instance(**init_kwargs)
        elif len(init_kwargs) > 0:
            if cls._init_instance(**init_kwargs) != cls.__instance:
                raise ValueError(
                    "Invalid configuration options specified. The global config "
                    f"was already initialized with values {cls.__instance}, but a "
                    "second initialization was requested with incompatible arguments "
                    f"{init_kwargs}."
                )
        return cls.__instance

    #: The imagenet validation path.
    imagenet_validation_path: str = "shifthappens/imagenet"

    #: The caching directory for model results (either absolute or relative to working directory). If the folder does not exist, it will be created.
    cache_directory_path: str = "shifthappens/cache"

    #: Show additional log messages on stderr (like progress bars).
    verbose: bool = False

    def __contains__(self, key):
        return key in self.__dict__


def get_config(**init_kwargs) -> Config:
    """
    Returns a global config initialized with provided arguments. This allows you to
    change defaults paths to ImageNet validation set, cached models result, etc.
    Note that reinitializing config will raise an error.
    For more details see :py:meth:`get_instance <shifthappens.config.Config.get_instance>`.
    Args:
        **init_kwargs: Values for initializing :py:class:`Config <shifthappens.config.Config>`
        fields.

    Returns:
        Initialized :py:class:`Config <shifthappens.config.Config>`.
    """
    return Config().get_instance(**init_kwargs)


def __getattr__(name: str) -> Any:
    config = get_config()
    if name in config:
        return getattr(config, name)
    raise AttributeError(name)


def __dir__() -> List[str]:
    config = get_config()
    return list(globals().keys()) + list(config.__dict__.keys())
