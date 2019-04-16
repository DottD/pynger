from pynger.types import Any
from memory_profiler import profile


def static_var(varname: str, value: Any):
    """ As a decorator, allows to define static variables.

    Args:
        varname: Name of the static variable
        value: Initial value of the new variable
        
    Example:
        >>> @static_var("counter", 0)
        >>> def foo():
        >>>     foo.counter += 1
        >>>     print(foo.counter)

    """
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate
