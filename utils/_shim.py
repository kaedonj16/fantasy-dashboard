"""Shared helper for compatibility shims.

When utils/* modules were consolidated, the old paths became re-export shims.
Tests (and some runtime code) monkeypatch attributes on the shim module, e.g.
``monkeypatch.setattr(utils.game_conditions, "fetch_week_odds", fake)``.
But the real functions live in the implementation module and resolve their
collaborators through the *implementation* module's globals, so patching the
shim alone has no effect.

``propagate_sets_to`` installs a module-level ``__setattr__``/``__delattr__``
that mirrors attribute sets/deletes onto the implementation module(s), making
the shim fully backward-compatible for monkeypatching.
"""
import sys


def propagate_sets_to(shim_name, *impls):
    """Make attribute sets/deletes on the shim propagate to impl modules.

    Args:
        shim_name: __name__ of the shim module (must already be in sys.modules).
        *impls: implementation module objects to mirror sets into. The first
            impl that already defines the attribute wins.
    """
    mod = sys.modules[shim_name]
    base_cls = type(mod)

    def __setattr__(self, name, value):
        base_cls.__setattr__(self, name, value)
        for impl in impls:
            if hasattr(impl, name):
                setattr(impl, name, value)
                break

    def __delattr__(self, name):
        base_cls.__delattr__(self, name)
        for impl in impls:
            if hasattr(impl, name):
                try:
                    delattr(impl, name)
                except AttributeError:
                    pass
                break

    mod.__class__ = type(
        "PropagatingModule",
        (base_cls,),
        {"__setattr__": __setattr__, "__delattr__": __delattr__},
    )
