import json
from typing import List, Dict, Tuple, Any, Callable, Iterable
from enum import Enum, unique
from io import StringIO


class StructLang(object):
    """
    自定义的表格描述标签语言
    """

    @unique
    class Placeholder(Enum):
        PAD = 0
        SOS = 1
        EOS = 2

    @unique
    class Vocab(Enum):
        LINE = 0
        CELL = 1
        HEAD = 2
        TAIL_H = 3
        TAIL_V = 4
        TAIL_T = 5

    def __init__(self, rows: int = 0, cols: int = 0, auto_init: bool = True):
        self._rows = rows
        self._cols = cols
        if rows > 0 and cols > 0 and auto_init:
            self._data = [([self.Vocab.CELL] * cols) for _ in range(rows)]

    def __setitem__(self, key: Tuple[int, int], value):
        self._data[key[0]][key[1]] = value

    def __getitem__(self, item: Tuple[int, int]):
        return self._data[item[0]][item[1]]

    def to_json(self, **kwargs):
        return json.dumps(self.to_object(), **kwargs)

    @classmethod
    def from_json(cls, s):
        return cls.from_object(json.loads(s))

    def to_object(self):
        return {
            'rows': self._rows,
            'cols': self._cols,
            'data': [[i.name for i in r] for r in self._data],
        }

    @classmethod
    def from_object(cls, obj_params):
        new_struct = cls(obj_params['rows'], obj_params['cols'], False)
        new_struct._data = [[
            getattr(cls.Vocab, i) for i in r] for r in obj_params['data']]
        return new_struct

    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    @property
    def size(self):
        return self._rows * self._cols + self._rows

    @property
    def data(self):
        return [[self._data[row][col] for col in range(self._cols)] for row in range(self._rows)]

    def cell(self, row: int, col: int):
        return self._data[row][col]

    @property
    def T(self):
        new_struct = self.__class__(self._cols, self._rows, False)
        new_struct._data = [list(i) for i in zip(*self._data)]
        mapping = {self.Vocab.TAIL_H: self.Vocab.TAIL_V, self.Vocab.TAIL_V: self.Vocab.TAIL_H}
        for row in range(new_struct._rows):
            for col in range(new_struct._cols):
                new_struct._data[row][col] = mapping.get(
                    new_struct._data[row][col], new_struct._data[row][col])
        return new_struct

    def copy(self):
        new_struct = self.__class__(self._rows, self._cols, False)
        new_struct._data = [[self._data[row][col] for col in range(self._cols)] for row in range(self._rows)]
        return new_struct

    def merge_cell(self, start: Tuple[int, int], end: Tuple[int, int]):
        assert 0 <= start[0] <= end[0] < self._rows
        assert 0 <= start[1] <= end[1] < self._cols
        assert start[0] < end[0] or start[1] < end[1]
        for row in range(start[0], end[0] + 1):
            for col in range(start[1], end[1] + 1):
                assert self._data[row][col] == self.Vocab.CELL
        self._data[start[0]][start[1]] = self.Vocab.HEAD
        for row in range(start[0] + 1, end[0] + 1):
            self._data[row][start[1]] = self.Vocab.TAIL_V
        for col in range(start[1] + 1, end[1] + 1):
            self._data[start[0]][col] = self.Vocab.TAIL_H
        for row in range(start[0] + 1, end[0] + 1):
            for col in range(start[1] + 1, end[1] + 1):
                self._data[row][col] = self.Vocab.TAIL_T

    def split_cell(self, pos: Tuple[int, int]):
        assert self._data[pos[0]][pos[1]] == self.Vocab.HEAD
        for row in range(pos[0] + 1, self._rows):
            if self._data[row][pos[1]] != self.Vocab.TAIL_V:
                break
            for col in range(pos[1] + 1, self._cols):
                if self._data[pos[0]][col] != self.Vocab.TAIL_H:
                    break
                assert self._data[row][col] == self.Vocab.TAIL_T
                self._data[row][col] = self.Vocab.CELL
        for row in range(pos[0] + 1, self._rows):
            if self._data[row][pos[1]] != self.Vocab.TAIL_V:
                break
            self._data[row][pos[1]] = self.Vocab.CELL
        for col in range(pos[1] + 1, self._cols):
            if self._data[pos[0]][col] != self.Vocab.TAIL_H:
                break
            self._data[pos[0]][col] = self.Vocab.CELL
        self._data[pos[0]][pos[1]] = self.Vocab.CELL

    def items(self):
        for row in range(self._rows):
            yield self.Vocab.LINE
            for col in range(self._cols):
                yield self._data[row][col]

    def flatten(self, offset=0):
        return [i.value + offset for i in self.items()]

    def labels(self, seq_len=0, use_sos=False, use_eos=False, cut_len=0):
        lab = self.flatten(len(self.Placeholder))  # 占位符在前
        if use_sos:
            lab = [self.Placeholder.SOS.value, *lab]
        if use_eos:
            lab = [*lab, self.Placeholder.EOS.value]
        if 0 < cut_len < len(lab): lab = lab[:cut_len]
        return lab + [
            self.Placeholder.PAD.value] * (seq_len - len(lab))

    def __str__(self):
        return '\n'.join(['\t'.join([self._data[row][col].name for col in range(
            self._cols)]) for row in range(self._rows)])

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_coordinate(cls, coordinate: List[Dict[str, int]],
                        keys=('startRowIndex', 'startColIndex', 'endRowIndex', 'endColIndex')):
        def is_merge_cell(n):
            return n[keys[0]] < n[keys[2]] or n[keys[1]] < n[keys[3]]

        table_rows = max([cell[keys[2]] for cell in coordinate]) + 1
        table_cols = max([cell[keys[3]] for cell in coordinate]) + 1

        new_struct = cls(table_rows, table_cols)

        for cell in coordinate:
            if is_merge_cell(cell):
                new_struct.merge_cell((cell[keys[0]], cell[keys[1]]), (cell[keys[2]], cell[keys[3]]))

        return new_struct

    def diff(self, other: Any, when_same: Callable[[str, str], Any] = lambda a, b: False,
             when_diff: Callable[[str, str], Any] = lambda a, b: True):
        if self._rows == other.rows and self._cols == other.cols:
            def content(row, col):
                return when_same(self._data[row][col].name, other.cell(row, col).name) \
                    if self._data[row][col] == other.cell(row, col) else when_diff(
                    self._data[row][col].name, other.cell(row, col).name)

            return [[content(row, col) for col in range(self._cols)] for row in range(self._rows)]

    def __eq__(self, other: Any):
        return self._rows == other.rows and self._cols == other.cols and self._data == other.data

    @classmethod
    def from_html(cls, html: str):
        pass

    def to_html(self, root_attr_callback: Callable[[], Iterable[Tuple[str, str]]] = None,
                cell_attr_callback: Callable[[int, int], Iterable[Tuple[str, str]]] = None,
                cell_text_callback: Callable[[int, int], str] = None, h_rows=0, i_cols=0):

        def cell_size(_r, _c):
            if self._data[_r][_c] == self.Vocab.CELL:
                return 1, 1
            else:
                rowspan, colspan = 1, 1
                if self._data[_r][_c] == self.Vocab.HEAD:
                    for r in range(_r + 1, self._rows):
                        if self._data[r][_c] != self.Vocab.TAIL_V:
                            break
                        rowspan += 1
                    for c in range(_c + 1, self._cols):
                        if self._data[_r][c] != self.Vocab.TAIL_H:
                            break
                        colspan += 1
                return rowspan, colspan

        def cell_attr(_r, _c):
            if callable(cell_attr_callback):
                for key, val in cell_attr_callback(_r, _c):
                    yield f' {key}="{val}"'
            rowspan, colspan = cell_size(_r, _c)
            if rowspan > 1:
                yield f' rowspan="{rowspan}"'
            if colspan > 1:
                yield f' colspan="{colspan}"'

        def root_attr():
            if callable(root_attr_callback):
                for key, val in root_attr_callback():
                    yield f' {key}="{val}"'

        def cell_name(_r, _c):
            return 'th' if _r < h_rows or _c < i_cols else 'td'

        with StringIO() as html:
            html.write('<table')
            html.write(''.join(root_attr()))
            html.write('>')
            for row in range(self._rows):
                html.write('<tr>')
                for col in range(self._cols):
                    if self._data[row][col] in (
                            self.Vocab.CELL, self.Vocab.HEAD):
                        html.write(f'<{cell_name(row, col)}')
                        html.write(''.join(cell_attr(row, col)))
                        html.write('>')
                        html.write(cell_text_callback(row, col))
                        html.write(f'</{cell_name(row, col)}>')
                html.write('</tr>')
            html.write('</table>')
            return html.getvalue()


if __name__ == '__main__':
    '''
    -------------------------------------------------------------------------
    |   CELL    |   HEAD       TAIL[H]  |   HEAD    |   HEAD        TAIL[H] |
    -------------------------------------           -                       -
    |   HEAD        TAIL[H] |   CELL    |  TAIL[V]  |  TAIL[V]      TAIL[+] |
    -------------------------------------------------------------------------
    '''
    coo = [
        {
            'startRowIndex': 0,
            'endRowIndex': 0,
            'startColIndex': 0,
            'endColIndex': 0,
        },
        {
            'startRowIndex': 0,
            'endRowIndex': 0,
            'startColIndex': 1,
            'endColIndex': 2,
        },
        {
            'startRowIndex': 0,
            'endRowIndex': 1,
            'startColIndex': 3,
            'endColIndex': 3,
        },
        {
            'startRowIndex': 0,
            'endRowIndex': 1,
            'startColIndex': 4,
            'endColIndex': 5,
        },
        {
            'startRowIndex': 1,
            'endRowIndex': 1,
            'startColIndex': 0,
            'endColIndex': 1,
        },
        {
            'startRowIndex': 1,
            'endRowIndex': 1,
            'startColIndex': 2,
            'endColIndex': 2,
        },
    ]

    st = StructLang(2, 6)
    st.merge_cell((0, 1), (0, 2))
    st.merge_cell((0, 3), (1, 3))
    st.merge_cell((0, 4), (1, 5))
    st.merge_cell((1, 0), (1, 1))
    print(st, '\n')

    t = StructLang.from_coordinate(coo)
    print(t, '\n')

    print(st.diff(t, lambda a, b: '', lambda a, b: f'{a}:{b}'), '\n')
    print(st == t, '\n')

    print(st.labels(20, True, False))
    print(st.labels(20, True, False, 15))
    print(st.labels(20, False, True))
    print(st.labels(20, False, True, 15))

    # s = StructLang(6, 6)
    # s.merge_cell((0, 1), (0, 5))
    # print(
    #     s.to_html(
    #         lambda: [('class', 'table'), ('width', '600'), ('height', '600')],
    #         lambda r, c: [('id', f'{r}-{c}')],
    #         lambda r, c: f'{r}-{c}',
    #         2, 1
    #     )
    # )
