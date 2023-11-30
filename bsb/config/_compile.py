import functools
from inspect import signature as _inspect_signature

_reserved_keywords = ["_parent", "_key"]


def _wrap_reserved(t):
    """
    Wrap a type handler in a wrapper that accepts all reserved keyword arguments that
    the config system will push into the type handler call, and pass only those that
    the original type handler accepts. This way type handlers can accept any
    combination of the reserved keyword args without raising TypeErrors when they do
    not accept one.
    """
    from .types import TypeHandler

    # Type handlers never need to be wrapped. The `__init_subclass__` of the TypeHandler
    # class handles wrapping of `__call__` implementations so that they accept and strip
    # _parent & _key.
    if isinstance(t, TypeHandler):
        return t

    # Check which reserved keywords the function already takes
    passes = _reserved_kw_passes(t)
    if all(passes.values()):
        return t

    # Create the keyword arguments of the outer function that accepts all reserved kwargs
    reserved_keys = "".join(f", {key}=None" for key in _reserved_keywords)
    header = f"def type_handler(value, *args{reserved_keys}, **kwargs):\n"
    passes = "".join(f", {key}={key}" for key in _reserved_keywords if passes[key])
    # Create the call to the inner function that is passed only the kwargs that it accepts
    wrap = f" return orig(value, *args{passes}, **kwargs)"
    # Compile the code block and indicate that the function was compiled here.
    mod = compile(header + wrap, f"{__file__}/<_wrap_reserved:compile>", "exec")
    # Execute the code block in this local scope and pick the function out of the scope
    exec(mod, {"orig": t}, bait := locals())
    type_handler = bait["type_handler"]
    # Copy over the metadata of the original function
    type_handler = functools.wraps(t)(type_handler)
    type_handler.__name__ = t.__name__
    return type_handler


def _reserved_kw_passes(f):
    # Inspect the signature and wrap the typecast in a wrapper that will accept and
    # strip the missing 'key' kwarg
    try:
        sig = _inspect_signature(f)
        params = sig.parameters
    except:
        params = []

    return {key: key in params for key in _reserved_keywords}
