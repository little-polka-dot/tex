
from contextlib import contextmanager
import re
from typing import List, Callable, Union, Tuple


class FormulaException(Exception):
    """ 表达式异常的父类 """


class UndefinedSymbol(FormulaException):
    """ 表达式包含未定义的符号 """


class EmptyStringMatched(FormulaException):
    """ 规则匹配到空字符串(错误的规则会导致死循环) """


class SymbolNotFound(FormulaException):
    """ 未找到匹配的符号 """


class RedundantStack(FormulaException):
    """ 表达式计算结束时发现多余的数值 """


class TooManyOperators(FormulaException):
    """ 多余的运算符 """


class Parentheses(object):

    class Head(object):

        def __init__(self, name):
            self._name = name

        def __str__(self): return self._name

        def __repr__(self):
            return f'Parentheses.Head("{self._name}")'

        @property
        def name(self): return self._name

    class Tail(object):

        def __init__(self, name, call):
            assert len(name) > 0
            self._name = name
            assert callable(call)
            self._call = call

        def __str__(self): return self._name

        def __repr__(self):
            return f'Parentheses.Tail("{self._name}")'

        @property
        def name(self): return self._name

        def __call__(self, *args): return self._call(*args)

    def __init__(self, head, tail, call):
        self._head = self.Head(head)
        self._tail = self.Tail(tail, call)

    @property
    def head(self): return self._head

    @property
    def tail(self): return self._tail

    def __str__(self): return f'{self._head}{self._tail}'

    def __repr__(self):
        return f'Parentheses("{self._head}", "{self._tail}")'

    def new_parentheses(self, call):
        return self.__class__(self.head.name, self.tail.name, call)


class Operator(object):

    def __init__(self, priority, name, call, n=0):
        assert n in (1, 2)
        self._operator_n = n
        assert priority > 0
        self._priority = priority
        assert len(name) > 0
        self._name = name
        assert callable(call)
        self._call = call

    def __str__(self): return self._name

    def __repr__(self):
        return f'Operator("{self._name}")'

    @property
    def name(self): return self._name

    @property
    def priority(self): return self._priority

    @property
    def n(self): return self._operator_n

    def new_operator(self, call):
        return self.__class__(self._priority, self._name, call)

    def __call__(self, *args): return self._call(*args)


class UnOperator(Operator):
    """ 单目运算符 示例：UnOperator('++', lambda x: x + 1) """

    def __init__(self, priority, name, call):
        super().__init__(priority, name, call, n=1)


class BiOperator(Operator):
    """ 双目运算符 示例：BiOperator(10, '+', lambda a, b: a + b) """

    def __init__(self, priority, name, call):
        super().__init__(priority, name, call, n=2)


class Formula(object):
    """
    >> Formula()('1+1')
    2
    >> Formula({'one': lambda s:1})('one+one')
    2
    """

    ValueRules = ((r'\d+\.\d*|\.\d+', float), (r'\d+', int))

    # 括号配置
    PaRegistry = (  # 三目及以上的运算符可以通过括号和逗号运算组合实现
        Parentheses('(', ')', lambda x: x),
        Parentheses('[', ']', lambda x: abs(x)),  # [-1] == 1
        # isnull(None) == True
        Parentheses('isnull(', ')', lambda x: True if x is None else False),
        # notnull(None) == False
        Parentheses('notnull(', ')', lambda x: True if x is not None else False),
        # if(conditions, a, b)
        Parentheses('if(', ')', lambda a, b, c: b if a else c),
        # 可以定义开始符号与结束符号相同的括号
        # Parentheses('||', '||', lambda x: abs(x)),
    )

    # 运算符配置
    # 由于匹配优先级的原因，如果在括号配置中定义了一个与运算符配置中某一条相同的符号，那么这个运算符配置将不会起作用！
    # 例如：如果定义了括号Parentheses('|', '|')，那么BiOperator('|')将会失效
    OpRegistry = (
        BiOperator(10, ',', lambda a, b: (*a, b) if isinstance(a, tuple) else (a, b)),
        BiOperator(20, '|', lambda a, b: a or b),
        BiOperator(21, '&', lambda a, b: a and b),
        UnOperator(22, '~', lambda x: not x),  # true~ == false
        BiOperator(25, '=', lambda a, b: a == b),
        BiOperator(25, '==', lambda a, b: a == b),
        BiOperator(25, '!=', lambda a, b: a != b),
        BiOperator(25, '>', lambda a, b: a > b),
        BiOperator(25, '>=', lambda a, b: a >= b),
        BiOperator(25, '<', lambda a, b: a < b),
        BiOperator(25, '<=', lambda a, b: a <= b),
        BiOperator(30, '+', lambda a, b: a + b),
        BiOperator(30, '-', lambda a, b: a - b),
        # UnOperator(31, '++', lambda x: x + 1),
        # UnOperator(31, '--', lambda x: x - 1),
        BiOperator(35, '*', lambda a, b: a * b),
        BiOperator(35, '/', lambda a, b: a / b),
        BiOperator(40, '?', lambda a, b: b if a is None else a),
    )

    def get_parentheses(self, name, __default=None):
        for i in self.pa_registry:
            if i.head.name == name:
                return i
        return __default  # 返回默认值

    def set_parentheses(self, n):
        for idx, i in enumerate(self.pa_registry):
            if i.head.name == n.head.name:
                self.pa_registry[idx] = n
                return i  # 返回旧运算符
        self.pa_registry.append(n)
        self.pa_registry.sort(
            key=lambda x: len(x.head.name), reverse=True)

    def del_parentheses(self, n):
        self.pa_registry.remove(n)
        self.pa_registry.sort(
            key=lambda x: len(x.head.name), reverse=True)

    def using_parentheses(self, n):
        """ 临时修改或临时添加一个运算规则 """
        @contextmanager
        def _with():
            o = self.set_parentheses(n)
            try:
                yield
            finally:
                if o: self.set_parentheses(o)
                else: self.del_parentheses(n)

        return _with()  # with _with()

    def get_operator(self, name, __default=None):
        for i in self.op_registry:
            if i.name == name:
                return i
        return __default  # 返回默认值

    def set_operator(self, n):
        for idx, i in enumerate(self.op_registry):
            if i.name == n.name:
                self.op_registry[idx] = n
                return i  # 返回旧运算符
        self.op_registry.append(n)
        self.op_registry.sort(
            key=lambda x: len(x.name), reverse=True)

    def del_operator(self, n):
        self.op_registry.remove(n)
        self.op_registry.sort(
            key=lambda x: len(x.name), reverse=True)

    def using_operator(self, n):
        """ 临时修改或临时添加一个运算规则 """
        @contextmanager
        def _with():
            o = self.set_operator(n)
            try:
                yield
            finally:
                if o: self.set_operator(o)
                else: self.del_operator(n)

        return _with()  # with _with()

    def __init__(self, rules: List[Tuple[str, Callable[[str], Union[int, float]]]] = None):
        """
        构建表达式解析器
        :param rules: 自定义模式 (正则表达式,固定格式的回调函数)
        """

        self.value_rules = [*(rules if rules else []), *self.ValueRules]
        self.pa_registry = sorted(self.PaRegistry, key=lambda x: len(x.head.name), reverse=True)
        self.op_registry = sorted(self.OpRegistry, key=lambda x: len(x.name), reverse=True)

    def leftmost_match(self, string, matched, pa_stack):
        """ 括号 > 运算符 > 自定义值匹配 > 默认值匹配 """

        # 可以根据上一个匹配到的值来判断当前位置是否满足括号头尾的条件
        head_matched = matched is None or \
            isinstance(matched, (BiOperator, Parentheses.Head))
        tail_matched = not head_matched

        if tail_matched and pa_stack:
            if string.startswith(pa_stack[-1].tail.name):
                i = pa_stack.pop()
                return i, string[len(i.tail.name):]

        if head_matched:
            for i in self.pa_registry:
                if string.startswith(i.head.name):
                    pa_stack.append(i)
                    return i.head, string[len(i.head.name):]

        for i in self.op_registry:
            if string.startswith(i.name):
                return i, string[len(i.name):]

        for p, i in self.value_rules:  # 匹配数值
            _matched = re.match(p, string)
            if _matched:
                value = _matched.group()
                if value:
                    return i(value), string[len(value):]
                else:
                    raise EmptyStringMatched(p)

        raise UndefinedSymbol(string)  # 未找到匹配的规则

    def match(self, string):
        matched, pa_stack = None, list()  # 临时存放括号
        while string:
            matched, string = self.leftmost_match(string, matched, pa_stack)
            yield matched
        if pa_stack:  # 发现未结束的括号
            raise SymbolNotFound(pa_stack[-1].tail.name)

    @classmethod
    def sort(cls, formula):

        # 1+2*(3-4)/5 -> 1234-)*5/+

        op_stack = list()  # 符号堆栈

        for u in formula:

            if isinstance(u, Parentheses.Head):
                op_stack.append(u)  # 括号头部

            elif isinstance(u, Parentheses):
                while op_stack and op_stack[-1] is not u.head:
                    yield op_stack.pop()
                if len(op_stack) < 1 or \
                        op_stack.pop() is not u.head:
                    raise SymbolNotFound(u.head.name)
                yield u.tail  # 括号运算为最高优先级

            elif isinstance(u, UnOperator):
                while op_stack and \
                        isinstance(op_stack[-1], BiOperator) and \
                        op_stack[-1].priority >= u.priority:
                    yield op_stack.pop()
                yield u  # 单目运算符不进堆栈

            elif isinstance(u, BiOperator):
                while op_stack and \
                        isinstance(op_stack[-1], BiOperator) and \
                        op_stack[-1].priority >= u.priority:
                    yield op_stack.pop()
                op_stack.append(u)  # 运算符

            else: yield u  # 值

        while op_stack: yield op_stack.pop()

    @classmethod
    def reduce(cls, formula):

        value_stack = list()

        def pop_n(stack, n):
            values = list()
            for _ in range(n): values.insert(0, stack.pop())
            return values

        for i in formula:

            if isinstance(i, Parentheses.Tail):
                if len(value_stack) < 1:
                    raise TooManyOperators(i.name)
                value = value_stack.pop()
                if isinstance(value, tuple):
                    value_stack.append(i(*value))
                else:
                    value_stack.append(i(value))

            elif isinstance(i, Operator):
                if len(value_stack) < i.n:
                    raise TooManyOperators(i.name)
                value_stack.append(
                    i(*pop_n(value_stack, i.n)))

            else:
                value_stack.append(i)

        if len(value_stack) != 1:
            raise RedundantStack(value_stack)
        else:
            return value_stack.pop()

    def __call__(self, formula_str: str):
        return self.reduce(self.sort(self.match(''.join(formula_str.split()))))


class OptionalFormula(Formula):

    def __init__(self, rules=None, allow_null=False):
        super().__init__(rules)
        if allow_null:
            self.set_operator(self.get_operator('+').new_operator(
                lambda a, b: b if a is None else (a if b is None else (a + b))))


if __name__ == '__main__':

    assert Formula()('1+2*(3-4)/5') == 0.6
    assert Formula()('1+2*((3-4))/5') == 0.6
    assert Formula()('1+2*[3-4]/5') == 1.4
    assert Formula()('1+2*[[3-4]]/5') == 1.4
    assert Formula()('1+2*([3-4])/5') == 1.4
    assert Formula([("n", lambda s: 5)])('1+2*(3-4)/n') == 0.6
    assert Formula()('3.0/2.0') == 1.5
    assert Formula()('1+2*([1+([0-1]+2)/(0-1)-1]-4)/5') == 0.6
    assert Formula()('(1+2)?0') == 3
    assert Formula([("n", lambda s: None)])('1+2*(3-4)/n?n?n?[0-5]') == 0.6
    f = Formula([("n", lambda s: None)])
    f.set_operator(f.get_operator('+').new_operator(lambda a, b: b if a is None else (a if b is None else (a + b))))
    assert f('1+2*(3-4)/(n+5)') == 0.6
    assert f('1+2*(3-4)/[n+(0-5)]') == 0.6
    assert f('n+5') == 5
    assert f('n?0-5') == -5
    assert f('n+(0-5)') == -5
    assert f('n?0-5') == -5
    # Formula([("n", lambda s: None)])('1+2*(3-4)/(n+5)')
    assert Formula()('1?2-2*5') == -9

    f = Formula([("n", lambda s: None)])
    with f.using_operator(f.get_operator('+').new_operator(lambda a, b: b if a is None else (a if b is None else (a + b)))):
        assert f('1+2*(3-4)/(n+5)') == 0.6

    assert Formula()('if(1<=0|1>0,1+2*(3-4)/5,0-5)') == 0.6

    f = Formula()
    f.set_operator(UnOperator(31, '++', lambda x: x + 1))
    assert f('2++*3') == 9
    assert f('3*2++') == 7
    assert f('1+1++*(3-1.5*2++)/(1++*2++)') == 0.6
