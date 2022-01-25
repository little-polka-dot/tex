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
            clip = (0, 0, self.W(page), self.H(page))
        clip = fitz.Rect(*clip)
        self._document[page].get_pixmap(
            matrix=mat, alpha=False, clip=clip).save(path)

    @staticmethod
    def in_clip(x0, y0, x1, y1, clip=None):
        return clip is None or (
                x0 >= clip[0] and y0 >= clip[1] and x1 <= clip[2] and y1 <= clip[3])

    def lines(self, page=0, film_mode=False, color_threshold=0.1,
              line_max_width=1, clip=None, filter_point=False):
        """ 尝试获取文档中所有线条(横竖两个方向)并返回其x0,y0,x1,y1坐标 """

        assert 0 <= color_threshold <= 1  # 颜色阈值条件在0到1之间

        def in_color(color):  # 是否满足颜色要求
            return (min(color) >= color_threshold) \
                if film_mode else (max(color) <= color_threshold)

        def is_line(width, length):  # 是否满足线条要求
            return (0 < width <= line_max_width < length) \
                if filter_point else (0 < width <= line_max_width)

        for path in self._document[page].get_drawings():
            for item in path['items']:
                if item[0] == 're':  # 矩形填充为线条
                    x0, y0, x1, y1 = item[1]
                    if self.in_clip(x0, y0, x1, y1, clip) \
                            and in_color(path['fill']) \
                            and is_line(min(x1 - x0, y1 - y0), max(x1 - x0, y1 - y0)):
                        yield x0, y0, x1, y1
                if item[0] == 'l':  # 直线
                    (x0, y0), (x1, y1) = item[1], item[2]
                    if x0 == x1 or y0 == y1:  # 过滤出横竖线
                        if self.in_clip(x0, y0, x1, y1, clip) \
                                and in_color(path['color']) \
                                and is_line(path['width'], (x1 - x0) + (y1 - y0)):
                            yield x0, y0, x1, y1

    def texts(self, page=0, return_text=False, clip=None):
        """ 尝试获取文档中所有文本并返回其x0,y0,x1,y1坐标 """
        for text in self._document[page].get_text("words"):
            if self.in_clip(*text[:4], clip):
                yield text[:5] if return_text else text[:4]

    def width(self, page=0):
        return self._document[page].rect.width

    w = W = width

    def height(self, page=0):
        return self._document[page].rect.height

    h = H = height

    def page_mask(self, page, bbox_list, fill=True):
        w, h = self.W(page), self.H(page)
        background = np.zeros((int(h), int(w)))
        for bbox in bbox_list:  # x0 y0 x1 y1
            x0, y0, x1, y1 = [int(i) for i in bbox]
            if fill:
                background[y0:y1+1, x0:x1+1] = 1
            else:
                background = cv2.rectangle(
                    background, (x0, y0), (x1, y1), 1)
        return background  # cv2.imwrite(...*255)


if __name__ == '__main__':
    with Loader('E:/Code/Mine/github/tex/test/pdf/89df2a78460636a6fa35edb53ade119b.pdf') as l:
        # debug_bbox(l.W(5), l.H(5), list(l.lines(5)) + list(l.texts(5)))
        l.screenshot(page=5)
