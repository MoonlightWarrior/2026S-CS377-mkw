from copy import deepcopy
from typing import Sequence, Type

from ..obs_builder import ObsBuilder

_obs_builders: dict[str, tuple[Type[ObsBuilder], dict | None]] = {}


def register_obs_builder(
    name: str | Sequence[str],
    cl: Type[ObsBuilder],
    default_kwargs: dict | None = None,
) -> None:
    if isinstance(name, str):
        _obs_builders[name] = (cl, default_kwargs)
    else:
        for n in name:
            _obs_builders[n] = (cl, default_kwargs)


def get_obs_builder(name: str, **additional_kwargs) -> ObsBuilder:
    cl, default_kwargs = _obs_builders[name]
    kwargs = deepcopy(default_kwargs) if default_kwargs else {}
    kwargs.update(additional_kwargs)
    return cl(**kwargs)
