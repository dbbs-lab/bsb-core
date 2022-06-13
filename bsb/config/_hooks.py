def _dunder(*hooks):
    return "__" + "_".join(hooks) + "__"


def run_hook(obj, hook, *args, **kwargs):
    """
    Execute the ``hook`` hook of ``obj``.

    Runs the ``hook`` method ``obj`` but also looks through the class hierarchy for
    essential hooks with the name ``__<hook>__``.

    .. note::

      Essential hooks are only ran if the method is called using ``run_hook`` while
      non-essential hooks are wrapped around the method and will always be executed when
      the method is called (see https://github.com/dbbs-lab/bsb/issues/158).
    """
    # Traverse the MRO tree and execute all essential (__hook__) methods.
    for parent in reversed(obj.__class__.__mro__):
        if hasattr(parent, _dunder(hook)):
            getattr(parent, _dunder(hook))(obj, *args, **kwargs)
    # Execute the final regular hook method.
    if hasattr(obj, hook):
        getattr(obj, hook)(*args, **kwargs)


def on(hook, cls, essential=False, before=False):
    """
    Register a class hook.

    :param hook: Name of the method to hook.
    :type hook: str
    :param cls: Class to hook.
    :type cls: type
    :param essential: If the hook is essential, it will always be executed even in child
      classes that override the hook. Essential hooks are only lost if the method on
      ``cls`` is replaced.
    :type essential: bool
    :param before: If ``before`` the hook is executed before the method, otherwise
      afterwards.
    :type before: bool
    """
    if essential:
        hook = _dunder(hook)
    return _hook(cls, hook, before)


def after(hook, cls, essential=False):
    """
    Register a class hook to run after the target method.

    :param hook: Name of the method to hook.
    :type hook: str
    :param cls: Class to hook.
    :type cls: type
    :param essential: If the hook is essential, it will always be executed even in child
      classes that override the hook. Essential hooks are only lost if the method on
      ``cls`` is replaced.
    :type essential: bool
    """
    return on(hook, cls, essential=essential, before=False)


def before(hook, cls, essential=False):
    """
    Register a class hook to run before the target method.

    :param hook: Name of the method to hook.
    :type hook: str
    :param cls: Class to hook.
    :type cls: type
    :param essential: If the hook is essential, it will always be executed even in child
      classes that override the hook. Essential hooks are only lost if the method on
      ``cls`` is replaced.
    :type essential: bool
    """
    return on(hook, cls, essential=essential, before=True)


def _super(cls, attr):
    for c in list(cls.__mro__)[1:]:
        try:
            return getattr(c, attr)
        except:
            pass


def has_hook(instance, hook):
    """
    Checks the existence of a method or essential method on the ``instance``.

    :param instance: Object to inspect.
    :type instance: object
    :param hook: Name of the hook to look for.
    :type hook: str
    """
    return hasattr(instance, hook) or hasattr(instance, _dunder(hook))


def overrides(cls, hook, mro=False):
    """
    Returns ``True`` if a class has implemented a method or ``False`` if it has inherited
    it.

    :param cls: Class to inspect.
    :type cls: class
    :param hook: Name of the hook to look for.
    :type hook: str
    """
    if not mro:
        return hook in vars(cls)
    else:

        class NotDefined:
            pass

        return getattr(cls, hook, NotDefined) is not getattr(object, hook, NotDefined)


def _hook(cls, hook, before):
    # Returns a decorator that will wrap the existing implementation if it exists,
    # otherwise returns a decorator that will simply add the decorated function to the
    # class.
    if overrides(cls, hook):
        return _make_wrapper(cls, hook, before)
    else:
        return _make_injector(cls, hook)


def _make_injector(cls, hook):
    def decorator(func):
        # Set the decorated function as the hooked function.
        _set_hooked_func(cls, hook, func)

    return decorator


def _get_func_to_hook(cls, hook):
    return getattr(cls, hook)


def _set_hooked_func(cls, hook, hooked):
    return setattr(cls, hook, hooked)


def _make_wrapper(cls, hook, before):
    func_to_hook = _get_func_to_hook(cls, hook)

    def decorator(func):
        if before:
            # Wrapper that executes the decorated function before the class method
            def hooked(*args, **kwargs):
                func(*args, **kwargs)
                r = func_to_hook(*args, **kwargs)
                return r

        else:
            # Wrapper that executes the decorated function after the class method
            def hooked(*args, **kwargs):
                r = func_to_hook(*args, **kwargs)
                func(*args, **kwargs)
                return r

        # Set the wrapper function as the hooked function
        _set_hooked_func(cls, hook, hooked)

    return decorator
