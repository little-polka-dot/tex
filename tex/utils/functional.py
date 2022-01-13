from functools import reduce
from typing import Iterable, Iterator


def call_or_pass(func, value):
    return func(value) if callable(func) else value


def and_(*__iter):
    if len(__iter) == 1 and isinstance(__iter[0], Iterable):
        return reduce(lambda a, b: a and b, __iter[0], True)
    else:
        return reduce(lambda a, b: a and b, __iter, True)


def or_(*__iter):
    if len(__iter) == 1 and isinstance(__iter[0], Iterable):
        return reduce(lambda a, b: a or b, __iter[0], False)
    else:
        return reduce(lambda a, b: a or b, __iter, False)


def mul(*__iter):
    if len(__iter) == 1 and isinstance(__iter[0], Iterable):
        return reduce(lambda a, b: a * b, __iter[0], 1)
    else:
        return reduce(lambda a, b: a * b, __iter, 1)


def is_odd(*__iter):
    if len(__iter) == 1 and isinstance(__iter[0], Iterable):
        return and_(map(lambda x: x % 2 != 0, __iter[0]))
    else:
        return and_(map(lambda x: x % 2 != 0, __iter))


def is_even(*__iter):
    if len(__iter) == 1 and isinstance(__iter[0], Iterable):
        return and_(map(lambda x: x % 2 == 0, __iter[0]))
    else:
        return and_(map(lambda x: x % 2 == 0, __iter))


def compare(func, a, b):
    return and_(map(lambda x: func(*x), zip(a, b)))


def is_gt(i, v):
    if isinstance(i, Iterable):
        return and_(map(lambda x: x > v, i))
    else:
        return i > v


def is_lt(i, v):
    if isinstance(i, Iterable):
        return and_(map(lambda x: x < v, i))
    else:
        return i < v


def is_eq(i, v):
    if isinstance(i, Iterable):
        return and_(map(lambda x: x == v, i))
    else:
        return i == v


def is_gte(i, v):
    if isinstance(i, Iterable):
        return and_(map(lambda x: x >= v, i))
    else:
        return i >= v


def is_lte(i, v):
    if isinstance(i, Iterable):
        return and_(map(lambda x: x <= v, i))
    else:
        return i <= v


def gt(i, v):
    if isinstance(i, Iterable):
        return or_(map(lambda x: x > v, i))
    else:
        return i > v


def lt(i, v):
    if isinstance(i, Iterable):
        return or_(map(lambda x: x < v, i))
    else:
        return i < v


def eq(i, v):
    if isinstance(i, Iterable):
        return or_(map(lambda x: x == v, i))
    else:
        return i == v


def gte(i, v):
    if isinstance(i, Iterable):
        return or_(map(lambda x: x >= v, i))
    else:
        return i >= v


def lte(i, v):
    if isinstance(i, Iterable):
        return or_(map(lambda x: x <= v, i))
    else:
        return i <= v


def map_(func, i):
    if isinstance(i, Iterable):
        if isinstance(i, Iterator):
            return map(func, i)
        else:
            return [func(x) for x in i]
    else:
        return func(i)


if __name__ == '__main__':
    print(compare(lambda a, b: a == b, [1, 2, 3], [1, 2, 4]))
