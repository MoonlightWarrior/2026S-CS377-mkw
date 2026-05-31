from copy import deepcopy
from typing import Sequence, Type

from ..action_parser import ActionParser

_action_parsers: dict[str, tuple[Type[ActionParser], dict | None]] = {}


def register_action_parser(
    name: str | Sequence[str],
    cl: Type[ActionParser],
    default_kwargs: dict | None = None,
) -> None:
    if isinstance(name, str):
        _action_parsers[name] = (cl, default_kwargs)
    else:
        for n in name:
            _action_parsers[n] = (cl, default_kwargs)


def get_action_parser(name: str, **additional_kwargs) -> ActionParser:
    cl, default_kwargs = _action_parsers[name]
    kwargs = deepcopy(default_kwargs) if default_kwargs else {}
    kwargs.update(additional_kwargs)
    return cl(**kwargs)
