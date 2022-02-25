# https://github.com/ArtifexSoftware/mupdf
# https://github.com/pymupdf/PyMuPDF
# https://pypi.org/project/PyMuPDF/

import fitz
import numpy as np
import cv2


class Loader(object):
    """ 坐标类型统一为 (x0, y0, x1, y1) """

    def __init__(self, path):
        self._document = fitz.Document(path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._document.close()

    def screenshot(self, path='screenshot.png', page=0, clip=None, ampl=(1, 1)):
        mat = fitz.Matrix(*ampl)  # 放大系数
        if clip is None:
            clip = (0, 0, self.w(page), self.h(page))
        clip = fitz.Rect(*clip)
        self._document[page].get_pixmap(
            matrix=mat, alpha=False, clip=clip).save(path)

    @staticmethod
    def in_clip(x0, y0, x1, y1, clip=None):
        return clip is None or (
                x0 >= clip[0] and y0 >= clip[1] and x1 <= clip[2] and y1 <= clip[3])

    def lines(self, page=0, film_mode=False, color_threshold=None,
              line_max_width=1, clip=None, combine_lines=False, line_combine_gap=1):
        """ 尝试获取文档中所有线条(横竖两个方向)并返回其x0,y0,x1,y1坐标 """

        assert color_threshold is None or 0 <= color_threshold <= 1

        def in_color(color):  # 是否满足颜色要求
            return color is not None and (color_threshold is None or
                ((max(color) >= color_threshold)
                    if film_mode else (min(color) <= color_threshold)))

        def lines_extractor():
            for path in self._document[page].get_drawings():
                for item in path['items']:
                    if item[0] == 'l':  # 直线
                        (x0, y0), (x1, y1) = item[1], item[2]
                        if x0 == x1 or y0 == y1:  # 过滤出横竖线
                            if x0 > x1 or y0 > y1:
                                x0, y0, x1, y1 = x1, y1, x0, y0
                            if self.in_clip(x0, y0, x1, y1, clip):
                                if 0 < path['width'] <= line_max_width and \
                                        in_color(path['color']):
                                    yield x0, y0, x1, y1
                    if item[0] == 're':  # 矩形填充为线条
                        x0, y0, x1, y1 = item[1]
                        if self.in_clip(x0, y0, x1, y1, clip):
                            if x1 >= x0 and y1 >= y0:  # 忽略掉无效矩形框
                                if min(x1 - x0, y1 - y0) <= line_max_width:
                                    if in_color(path['color']) or in_color(path['fill']):
                                        yield x0, y0, x1, y1
                                else:
                                    if in_color(path['color']) or \
                                            path['color'] is None and path['fill']:
                                        # 具有填充颜色的矩形区域的四条边也作为有效的表格线条
                                        # TODO 边框颜色不满足条件时填充颜色也需要条件筛选
                                        yield x0, y0, x0 + path['width'], y1
                                        yield x0, y0, x1, y0 + path['width']
                                        yield x0, y1, x1, y1 + path['width']
                                        yield x1, y0, x1 + path['width'], y1

        def try_connect(line1, line2):
            if abs(line1[0] - line2[0]) <= line_combine_gap and \
                    abs(line1[2] - line2[2]) <= line_combine_gap:
                x0 = min(line2[0], line1[0])
                y0 = min(line2[1], line1[1])
                x1 = max(line2[2], line1[2])
                y1 = max(line2[3], line1[3])
                sum_len = (line2[3] - line2[1]) + (line1[3] - line1[1])
                if y1 - y0 <= sum_len + line_max_width:
                    return x0, y0, x1, y1
            if abs(line1[1] - line2[1]) <= line_combine_gap and \
                    abs(line1[3] - line2[3]) <= line_combine_gap:
                x0 = min(line2[0], line1[0])
                y0 = min(line2[1], line1[1])
                x1 = max(line2[2], line1[2])
                y1 = max(line2[3], line1[3])
                sum_len = (line2[2] - line2[0]) + (line1[2] - line1[0])
                if x1 - x0 <= sum_len + line_max_width:
                    return x0, y0, x1, y1

        def connect_head(line_list):
            line = line_list.pop()
            line_dump = list()
            while line_list:
                for line_ in line_list:
                    new_line = try_connect(line_, line)
                    if new_line: line = new_line
                    else: line_dump.append(line_)
                if len(line_dump) >= len(line_list):
                    break  # 没有找到相接的线
                else:
                    line_list = line_dump
                    line_dump = list()
            return line, line_list

        if combine_lines:
            lines = [*lines_extractor()]
            while lines:
                head, lines = connect_head(lines)
                yield head
        else:
            yield from lines_extractor()

    def texts(self, page=0, return_text=False, clip=None):
        """ 尝试获取文档中所有文本并返回其x0,y0,x1,y1坐标 """
        for text in self._document[page].get_text("words"):
            if self.in_clip(*text[:4], clip):
                yield text[:5] if return_text else text[:4]

    def to(self, name, page=0):
        """
        name: html | json | rawjson | dict | rawdict | xml | xhtml
        """
        return self._document[page].get_text(name)

    def width(self, page=0):
        return self._document[page].rect.width

    w = W = width

    def height(self, page=0):
        return self._document[page].rect.height

    h = H = height

    def mask(self, page, bbox_list, fill=True):
        w, h = self.w(page), self.h(page)
        background = np.zeros((int(h), int(w)))
        for bbox in bbox_list:  # x0 y0 x1 y1
            x0, y0, x1, y1 = [int(i) for i in bbox]
            if fill:
                background[y0:y1 + 1, x0:x1 + 1] = 1
            else:
                background = cv2.rectangle(
                    background, (x0, y0), (x1, y1), 1)
        return background  # cv2.imwrite(...*255)


if __name__ == '__main__':
    with Loader(r'E:\Data\source\pdf\广发证券：2021年半年度报告.pdf') as l:
        # debug_bbox(l.W(5), l.H(5), list(l.lines(5)) + list(l.texts(5)))
        page = 62
        for i in l.texts(page, return_text=True):
            print(i[-1])
        # a1 = [*l.lines(page, line_max_width=2, combine_lines=True, line_combine_gap=1)]
        # a2 = [*l.texts(page)]
        # # for i in range(len(a1)):
        # cv2.imshow('2', l.mask(page, [*a2]))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
