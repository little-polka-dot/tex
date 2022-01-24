# https://github.com/ArtifexSoftware/mupdf
# https://github.com/pymupdf/PyMuPDF
# https://pypi.org/project/PyMuPDF/


class Loader(object):

    def __init__(self, path):
        from fitz import Document  # 非必须安装项
        self._document = Document(path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._document.close()

    def lines(self, page=0, color_threshold=0.0, reverse=False, line_max_width=2):
        assert 0 <= color_threshold <= 1
        doc_page = self._document[page]
        for path in doc_page.get_drawings():
            if reverse:
                for item in path['items']:
                    if item[0] == 're':  # 矩形填充为线条
                        x0, y0, x1, y1 = item[1]
                        if min(path['fill']) >= color_threshold \
                                and 0 < min(x1-x0, y1-y0) <= line_max_width:
                            yield x0, y0, x1, y1
                    if item[0] == 'l':
                        (x0, y0), (x1, y1) = item[1], item[2]
                        if x0 == x1 or y0 == y1:  # 过滤出横竖线
                            if min(path['color']) >= color_threshold \
                                    and 0 < path["width"] <= line_max_width:
                                yield x0, y0, x1, y1
            else:
                for item in path['items']:
                    if item[0] == 're':  # 矩形填充为线条
                        x0, y0, x1, y1 = item[1]
                        if max(path['fill']) <= color_threshold \
                                and 0 < min(x1-x0, y1-y0) <= line_max_width:
                            yield x0, y0, x1, y1
                    if item[0] == 'l':
                        (x0, y0), (x1, y1) = item[1], item[2]
                        if x0 == x1 or y0 == y1:  # 过滤出横竖线
                            if max(path['color']) <= color_threshold \
                                    and 0 < path["width"] <= line_max_width:
                                yield x0, y0, x1, y1

    def texts(self, page=0, return_text=False):
        doc_page = self._document[page]
        for text in doc_page.get_text("words"):
            x0, y0, x1, y1, st = text[:5]
            if return_text:
                yield x0, y0, x1, y1, st
            else:
                yield x0, y0, x1, y1

    def width(self, page=0):
        return self._document[page].rect.width

    w = W = width

    def height(self, page=0):
        return self._document[page].rect.height

    h = H = height


def debug_bbox(w, h, bbox_list, fill=True):
    import numpy as np
    import cv2 as cv
    background = np.zeros((int(h), int(w)))
    for bbox in bbox_list:  # x0 y0 x1 y1
        if fill:
            background[int(bbox[1]):int(bbox[3])+1, int(bbox[0]):int(bbox[2])+1] = 1
        else:
            background = cv.rectangle(
                background, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 1)
    cv.imshow('', background)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    with Loader('E:/Code/Mine/github/tex/test/pdf/89df2a78460636a6fa35edb53ade119b.pdf') as l:
        debug_bbox(l.W(5), l.H(5), list(l.lines(5)) + list(l.texts(5)))