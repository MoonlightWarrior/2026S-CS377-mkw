from copy import deepcopy
from typing import Sequence, Type

from ..reward_function import RewardFunction

_reward_functions: dict[str, tuple[Type[RewardFunction], dict | None]] = {}


def register_reward_function(
    name: str | Sequence[str],
    cl: Type[RewardFunction],
    default_kwargs: dict | None = None,
) -> None:
    if isinstance(name, str):
        _reward_functions[name] = (cl, default_kwargs)
    else:
        for n in name:
            _reward_functions[n] = (cl, default_kwargs)


def get_reward_function(name: str, **additional_kwargs) -> RewardFunction:
    cl, default_kwargs = _reward_functions[name]
    kwargs = deepcopy(default_kwargs) if default_kwargs else {}
    kwargs.update(additional_kwargs)
    return cl(**kwargs)
