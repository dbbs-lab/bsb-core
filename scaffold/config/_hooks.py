def _hook(obj, hook, callback, essential, *args, **kwargs):
    pass


def _dunder(*hooks):
    return "__" + "_".join(hooks) + "__"


def run_hook(obj, hook, *args, **kwargs):
    print("looking for", _dunder(hook), "on", obj.__class__.__name__)
    for parent in reversed(obj.__class__.__mro__):
        if hasattr(parent, _dunder(hook)):
            print("Found it on", parent.__class__.__name__)
            getattr(parent, _dunder(hook))(obj, *args, **kwargs)
    if hasattr(obj, hook):
        getattr(obj, hook)(*args, **kwargs)


def on(hook, cls, essential=False):
    if essential:
        hook = _dunder(hook)
    return _hook(hook, cls, essential)

    _prepare_class_hook(cls, hook)

    def inner_decorator(func):
        def wrapper(*args, **kwargs):
            r = func(*args, **kwargs)


def _super(cls, attr):
    for c in list(cls.__mro__)[1:]:
        try:
            return getattr(c, attr)
        except:
            pass
    else:
        raise AttributeError(
            "Attribute {} not found in mro of {}".format(attr, cls.__name__)
        )


def _hook(hook, cls, essential):
    if not hasattr(cls, hook) or getattr(cls, hook) is _super(cls, hook):
        return _make_injector(hook, cls)
    else:
        return _make_wrapper(hook, cls)


def _make_injector(hook, cls):
    def injector(func):
        setattr(cls, hook, func)

    return injector


def _make_wrapper(hook, cls):
    pass
