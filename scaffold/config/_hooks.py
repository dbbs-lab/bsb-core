def _dunder(*hooks):
    return "__" + "_".join(hooks) + "__"


def run_hook(obj, hook, *args, **kwargs):
    for parent in reversed(obj.__class__.__mro__):
        if hasattr(parent, _dunder(hook)):
            print("Found " + hook + "it on", parent.__class__.__name__)
            getattr(parent, _dunder(hook))(obj, *args, **kwargs)
    if hasattr(obj, hook):
        getattr(obj, hook)(*args, **kwargs)


def on(hook, cls, essential=False, before=False):
    if essential:
        hook = _dunder(hook)
    return _hook(cls, hook, before)


def after(hook, cls, essential=False):
    return on(hook, cls, essential=essential, before=False)


def before(hook, cls, essential=False):
    return on(hook, cls, essential=essential, before=True)


def _super(cls, attr):
    for c in list(cls.__mro__)[1:]:
        try:
            return getattr(c, attr)
        except:
            pass


def has_hook(instance, hook):
    return hasattr(instance, hook) or hasattr(instance, _dunder(hook))


def overrides(cls, hook):
    return hasattr(cls, hook) and getattr(cls, hook) is not _super(cls, hook)


def _hook(cls, hook, before):
    if overrides(cls, hook):
        return _make_wrapper(cls, hook, before)
    else:
        return _make_injector(cls, hook)


def _make_injector(cls, hook):
    def decorator(func):
        setattr(cls, hook, func)

    return decorator


def _get_func_to_hook(cls, hook):
    return getattr(cls, hook)


def _set_hooked_func(cls, hook, hooked):
    return setattr(cls, hook, hooked)


def _make_wrapper(cls, hook, before):
    func_to_hook = _get_func_to_hook(cls, hook)

    def decorator(func):
        if before:

            def hooked(*args, **kwargs):
                func(*args, **kwargs)
                r = func_to_hook(*args, **kwargs)
                return r

        else:

            def hooked(*args, **kwargs):
                r = func_to_hook(*args, **kwargs)
                func(*args, **kwargs)
                return r

        _set_hooked_func(cls, hook, hooked)

    return decorator
