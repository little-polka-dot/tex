import random
from enum import Enum, unique
from typing import Dict
from tex.data.labels.pipeline import StructLang


@unique
class BorderType(Enum):
    TopBorder = 0  # 行-上边框
    BottomBorder = 1  # 行-下边框
    LeftBorder = 2  # 列-左边框
    RightBorder = 3  # 列-右边框
    IndexRow = 4  # 索引区域行分割线
    IndexCol = 5  # 索引区域列分割线
    ContentRow = 6  # 内容区域行分割线
    ContentCol = 7  # 内容区域列分割线
    HeaderRow = 8  # 字段区域行分割线
    HeaderCol = 9  # 字段区域列分割线
    IndexContent = 10  # 索引-内容区域分割线
    HeaderContent = 11  # 字段-内容区域分割线
    RootRow = 12  # 左上角区域行分割线
    RootCol = 13  # 左上角区域列分割线
    RootHeader = 14  # 左上角-字段区域分割线
    RootIndex = 15  # 左上角-索引区域分割线


@unique
class CellType(Enum):
    Root = 0
    Header = 1
    Index = 2
    Content = 3


def random_color(seg: int = 1, seg_alpha: int = None):
    """
    seg 颜色区域分段
    seg_alpha 当给定该数值时 从该数值指定的颜色段获取随机颜色 反之则从颜色段边界中随机获取颜色
    self.random_color(2, 2) if dark else self.random_color(2, 1)
    """

    def to_rgb(x):
        t = [x[b] * (256 ** b) for b in range(3)]
        s = hex(sum(t))[2:].upper()
        return '#' + '0' * (6 - len(s)) + s

    if seg_alpha:
        assert 1 <= seg_alpha <= seg
        seg_a = (256 // seg) * (seg_alpha - 1)
        seg_b = (256 // seg) * seg_alpha - 1
        chn_c = random.randint(seg_a, seg_b)
        return to_rgb([chn_c] * 3)
    else:
        assert seg > 0
        colors = [to_rgb([0] * 3), to_rgb([255] * 3)]
        if seg > 1:
            for i in range(seg - 1):
                colors.append(
                    to_rgb([(i + 1) * (256 // seg)] * 3))
        return random.choice(colors)


def random_border_width(a=1, b=5):
    return f'{random.randint(a, b)}px'


def random_border_style():
    return random.choice(['dotted', 'dashed', 'solid'])


class BorderStyle(object):

    def __init__(self, width, style, color):
        self._width = width
        self._style = style
        self._color = color

    def __str__(self):
        return f'{self._width} {self._style} {self._color}'

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__str__()})'

    def __eq__(self, other):
        return self._width == other.border_width and \
               self._style == other.border_style and self._color == other.color

    def __hash__(self):
        return hash(f'{self._width} {self._style} {self._color}')


class RandomStyle(object):
    """
        用于生成随机的CSS样式
        table {...}
        table td {...}
    """

    def row_col_iterator(self, row):
        for col in range(0, self._struct.cols):
            if self._struct.cell(row, col) in (StructLang.Vocab.CELL, StructLang.Vocab.HEAD):
                n = len([i for i in range(0, col + 1) if self._struct.cell(row, i) in (
                    StructLang.Vocab.CELL, StructLang.Vocab.HEAD)])
                yield f'table tr:nth-child({row + 1}) *:nth-child({n})'

    def col_row_iterator(self, col):
        for row in range(0, self._lang.rows):
            if self._lang.cell(row, col) in (StructLang.Vocab.CELL, StructLang.Vocab.HEAD):
                n = len([i for i in range(0, row + 1) if self._lang.cell(i, col) in (
                    StructLang.Vocab.CELL, StructLang.Vocab.HEAD)])

    def __init__(self, struct: StructLang, h_rows=0, i_cols=0):
        self._struct = struct
        self._h_rows = h_rows
        self._i_cols = i_cols

    def generate(self, border_dict: Dict[BorderType, BorderStyle]):
        for css in self.generate_border(border_dict):
            yield css
        # TODO cell_dict background...

    def generate_border(self, border_dict: Dict[BorderType, BorderStyle]):
        yield 'table { border-collapse:collapse; }'
        if BorderType.TopBorder in border_dict:
            yield f'table {{ border-top: {border_dict[BorderType.TopBorder]}; }}'
        if BorderType.BottomBorder in border_dict:
            yield f'table {{ border-bottom: {border_dict[BorderType.BottomBorder]}; }}'
        if BorderType.LeftBorder in border_dict:
            yield f'table {{ border-left: {border_dict[BorderType.LeftBorder]}; }}'
        if BorderType.RightBorder in border_dict:
            yield f'table {{ border-right: {border_dict[BorderType.RightBorder]}; }}'
        if BorderType.HeaderContent in border_dict:
            if self._h_rows > 0:
                for i in range(self._t_cols - self._i_cols):
                    r_css = f'tr:nth-child({self._h_rows})'
                    c_css = f'*:nth-child({i + 1 + self._i_cols})'
                    value = f'border-bottom: {border_dict[BorderType.HeaderContent]}'
                    yield f'table {r_css} {c_css} {{ {value}; }}'
        if BorderType.IndexContent in border_dict:
            if self._i_cols > 0:
                for i in range(self._t_rows - self._h_rows):
                    r_css = f'tr:nth-child({i + 1 + self._h_rows})'
                    c_css = f'*:nth-child({self._i_cols})'
                    value = f'border-right: {border_dict[BorderType.IndexContent]}'
                    yield f'table {r_css} {c_css} {{ {value}; }}'
        if BorderType.IndexRow in border_dict:
            for i in range(self._t_rows - self._h_rows - 1):
                r_css = f'tr:nth-child({i + self._h_rows + 1})'
                for n in range(self._i_cols):
                    c_css = f'*:nth-child({n + 1})'
                    values = f'border-bottom: {border_dict[BorderType.IndexRow]}'
                    yield f'table {r_css} {c_css} {{ {values}; }}'


if __name__ == '__main__':
    BorderStyle(
        random_border_width(),
        random_border_style(),
        random_color(2, 1)
    )

