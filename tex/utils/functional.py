from functools import reduce
from typing import Iterable


def mul(*__iter):
    return reduce(lambda a, b: a * b, __iter[0], 1) \
        if len(__iter) == 1 and isinstance(__iter[0], Iterable) \
        else reduce(lambda a, b: a * b, __iter, 1)


def all_odd(*__iter):
    return all(map(lambda x: x % 2 != 0, __iter[0])) \
        if len(__iter) == 1 and isinstance(__iter[0], Iterable) \
        else all(map(lambda x: x % 2 != 0, __iter))


def all_even(*__iter):
    return all(map(lambda x: x % 2 == 0, __iter[0])) \
        if len(__iter) == 1 and isinstance(__iter[0], Iterable) \
        else all(map(lambda x: x % 2 == 0, __iter))


def compare(func, a, b):
    return all(map(lambda x: func(*x), zip(a, b)))


def all_gt(i, v):
    return all(map(lambda x: x > v, i)) if isinstance(i, Iterable) else i > v


def all_lt(i, v):
    return all(map(lambda x: x < v, i)) if isinstance(i, Iterable) else i < v


def all_eq(i, v):
    return all(map(lambda x: x == v, i)) if isinstance(i, Iterable) else i == v


def all_gte(i, v):
    return all(map(lambda x: x >= v, i)) if isinstance(i, Iterable) else i >= v


def all_lte(i, v):
    return all(map(lambda x: x <= v, i)) if isinstance(i, Iterable) else i <= v


def any_gt(i, v):
    return any(map(lambda x: x > v, i)) if isinstance(i, Iterable) else i > v


def any_lt(i, v):
    return any(map(lambda x: x < v, i)) if isinstance(i, Iterable) else i < v


def any_eq(i, v):
    return any(map(lambda x: x == v, i)) if isinstance(i, Iterable) else i == v


def any_gte(i, v):
    return any(map(lambda x: x >= v, i)) if isinstance(i, Iterable) else i >= v


def any_lte(i, v):
    return any(map(lambda x: x <= v, i)) if isinstance(i, Iterable) else i <= v


def map_(func, i):
    return map(func, i) if isinstance(i, Iterable) else func(i)


def list_(i):
    return list(i) if isinstance(i, Iterable) else i


if __name__ == '__main__':
    print(any_eq([0, 1, 1], 0))
    print(mul(0, 1, 2, 3))
