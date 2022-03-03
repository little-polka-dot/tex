import re
from typing import Dict, Callable, Union
from math import pi, e, inf, nan
import operator


class FormulaException(Exception): pass


class UndefinedSymbol(FormulaException): pass


class Formula(object):
    """
    >> Formula()('1+2*(3-4)/5')
    0.6
    >> Formula({r"five":lambda s:5})('1+2*(3-4)/five')
    0.6
    """

    def __init__(self, mapping: Dict[str, Callable[[str], Union[int, float]]] = None):
        """
        构建表达式解析器
        :param mapping: 自定义模式 key为正则表达式 value为固定格式的回调函数
        """
        if mapping is None: mapping = {}
        self.mapping = {
            r'(\d+(\.\d*)?)|(\.\d+)': lambda s: float(s),
            r'\[(-1|pi|e)\]': lambda s: {
                '-1': -1, 'pi': pi, 'e': e}[s.strip('[]')],
            **mapping
        }

    def __call__(self, formula_str: str):

        formula_format = ''.join(formula_str.split())

        def match(string):
            for pattern, callback in self.mapping.items():
                result = re.match(pattern, string)
                if result:
                    return (
                        callback(result.group()),
                        result.group(),
                        string[len(result.group()):]
                    )
            raise UndefinedSymbol(string)

        ops = {'+', '-', '*', '/', '(', ')'}
        formula_list = list()  # 中序列表
        while formula_format:
            if formula_format[0] in ops:
                formula_list.append(formula_format[0])
                formula_format = formula_format[1:]
            else:
                value, matched, formula_format = match(formula_format)
                formula_list.append(value)

        op_stack = list()
        po_order = list()
        for u in formula_list:
            if u in ops:
                if u in ('(',):
                    op_stack.append(u)
                if u in (')',):
                    while op_stack and op_stack[-1] != '(':
                        po_order.append(op_stack.pop())
                    assert op_stack.pop() == '('
                if u in ('+', '-'):
                    while op_stack and op_stack[-1] in ('*', '/', '+', '-'):
                        po_order.append(op_stack.pop())
                    op_stack.append(u)
                if u in ('*', '/'):
                    while op_stack and op_stack[-1] in ('*', '/'):
                        po_order.append(op_stack.pop())
                    op_stack.append(u)
            else:
                po_order.append(u)
        while op_stack:
            po_order.append(op_stack.pop())

        op_dict = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
        }
        po_stack = list()
        for i in po_order:
            if i in op_dict:
                r_val = po_stack.pop()
                l_val = po_stack.pop()
                try:
                    v = op_dict[i](l_val, r_val)
                except ZeroDivisionError:
                    v = inf  # .../0 == inf
                po_stack.append(v)
            else:
                po_stack.append(i)

        assert len(po_stack) == 1

        if po_stack:
            return po_stack.pop()


if __name__ == '__main__':
    print(Formula()('1+2*(3-4)/5'))
