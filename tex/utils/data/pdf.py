# https://github.com/ArtifexSoftware/mupdf
# https://github.com/pymupdf/PyMuPDF
# https://pypi.org/project/PyMuPDF/

import fitz
import numpy as np
import cv2


class Loader(object):

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

    def lines(self, page=0, color_threshold=0.0, reverse=False, line_max_width=2, clip=None):
        def in_line(color, width):
            if reverse:
                return min(color) >= color_threshold \
                       and 0 < width <= line_max_width
            else:
                return max(color) <= color_threshold \
                       and 0 < width <= line_max_width
        assert 0 <= color_threshold <= 1
        for path in self._document[page].get_drawings():
            for item in path['items']:
                if item[0] == 're':  # 矩形填充为线条
                    x0, y0, x1, y1 = item[1]
                    if self.in_clip(x0, y0, x1, y1, clip) and \
                            in_line(path['fill'], min(x1-x0, y1-y0)):
                        yield x0, y0, x1, y1
                if item[0] == 'l':  # 直线
                    (x0, y0), (x1, y1) = item[1], item[2]
                    if x0 == x1 or y0 == y1:  # 过滤出横竖线
                        if self.in_clip(x0, y0, x1, y1, clip) and \
                                in_line(path['color'], path['width']):
                            yield x0, y0, x1, y1

    def texts(self, page=0, return_text=False, clip=None):
        for text in self._document[page].get_text("words"):
            if self.in_clip(*text[:4], clip):
                yield text[:5] if return_text else text[:4]

    def width(self, page=0):
        return self._document[page].rect.width

    w = W = width

    def height(self, page=0):
        return self._document[page].rect.height

    h = H = height

    def bbox2mask(self, page, bbox_list, fill=True):
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
