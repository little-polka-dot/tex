from functools import reduce
from typing import Iterable


def optional_function(func, default=None):
    """
    >> func: Callable
    >> func_default: Callable
    >> optional_function(func, func_default) == func
    True
    >> func = None
    >> optional_function(func, func_default) == func_default
    True
    """
    if default is None: default = lambda *w, **kw: None
    assert callable(func) or callable(default)
    return func if callable(func) else default


def and_(*__iter):
    return reduce(lambda a, b: a and b, __iter[0], True) \
        if len(__iter) == 1 and isinstance(__iter[0], Iterable) \
        else reduce(lambda a, b: a and b, __iter, True)


def or_(*__iter):
    return reduce(lambda a, b: a or b, __iter[0], False) \
        if len(__iter) == 1 and isinstance(__iter[0], Iterable) \
        else reduce(lambda a, b: a or b, __iter, False)


def mul(*__iter):
    return reduce(lambda a, b: a * b, __iter[0], 1) \
        if len(__iter) == 1 and isinstance(__iter[0], Iterable) \
        else reduce(lambda a, b: a * b, __iter, 1)


def is_odd(*__iter):
    return and_(map(lambda x: x % 2 != 0, __iter[0])) \
        if len(__iter) == 1 and isinstance(__iter[0], Iterable) \
        else and_(map(lambda x: x % 2 != 0, __iter))


def is_even(*__iter):
    return and_(map(lambda x: x % 2 == 0, __iter[0])) \
        if len(__iter) == 1 and isinstance(__iter[0], Iterable) \
        else and_(map(lambda x: x % 2 == 0, __iter))


def compare(func, a, b):
    return and_(map(lambda x: func(*x), zip(a, b)))


def is_gt(i, v):
    return and_(map(lambda x: x > v, i)) if isinstance(i, Iterable) else i > v


def is_lt(i, v):
    return and_(map(lambda x: x < v, i)) if isinstance(i, Iterable) else i < v


def is_eq(i, v):
    return and_(map(lambda x: x == v, i)) if isinstance(i, Iterable) else i == v


def is_gte(i, v):
    return and_(map(lambda x: x >= v, i)) if isinstance(i, Iterable) else i >= v


def is_lte(i, v):
    return and_(map(lambda x: x <= v, i)) if isinstance(i, Iterable) else i <= v


def gt(i, v):
    return or_(map(lambda x: x > v, i)) if isinstance(i, Iterable) else i > v


def lt(i, v):
    return or_(map(lambda x: x < v, i)) if isinstance(i, Iterable) else i < v


def eq(i, v):
    return or_(map(lambda x: x == v, i)) if isinstance(i, Iterable) else i == v


def gte(i, v):
    return or_(map(lambda x: x >= v, i)) if isinstance(i, Iterable) else i >= v


def lte(i, v):
    return or_(map(lambda x: x <= v, i)) if isinstance(i, Iterable) else i <= v


def map_(func, i):
    return map(func, i) if isinstance(i, Iterable) else func(i)


def list_(i):
    return list(i) if isinstance(i, Iterable) else i


if __name__ == '__main__':
    print(compare(lambda a, b: a == b, [1, 2, 3], [1, 2, 4]))
