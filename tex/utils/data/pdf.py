# from https://github.com/euske/pdfminer

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.layout import LTChar, LTTextLine, LTTextBox, LTLine, LTRect, LTCurve, LTItem


def show_bbox(w, h, bbox_list):
    import numpy as np
    import cv2 as cv
    background = np.zeros((int(h), int(w)))
    for bbox in bbox_list:  # x-min y-min x-max y-max
        background = cv.rectangle(
            background,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            1
        )
    cv.imshow('', background)
    cv.waitKey(0)
    cv.destroyAllWindows()


def pdf2bbox(path, password=b''):
    with open(path, 'rb') as fp:
        # Create a PDF parser object associated with the file object.
        parser = PDFParser(fp)
        # Create a PDF document object that stores the document structure.
        # Supply the password for initialization.
        document = PDFDocument(parser, password)
        # Check if the document allows text extraction. If not, abort.
        if not document.is_extractable:
            raise PDFTextExtractionNotAllowed
        # Create a PDF resource manager object that stores shared resources.
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        # Create a PDF page aggregator object.
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        # Create a PDF interpreter object.
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        # Process each page contained in the document.
        for num_page, page in enumerate(PDFPage.create_pages(document), start=1):
            w, h = [int(i) for i in page.attrs['MediaBox'][2:4]]
            interpreter.process_page(page)
            # a_1 = [x for x in device.get_result() if isinstance(x, LTChar)]
            # a_2 = [x for x in device.get_result() if isinstance(x, LTTextBox)]
            # a_3 = [x for x in device.get_result() if isinstance(x, LTLine)]
            a_4 = [x for x in device.get_result() if isinstance(x, LTRect)]
            show_bbox(w, h, [x.bbox for x in device.get_result() if isinstance(x, LTRect) and (x.height < 2)])


# https://github.com/ArtifexSoftware/mupdf
# https://github.com/pymupdf/PyMuPDF
# https://pypi.org/project/PyMuPDF/

import fitz
from tex.utils.functional import all_gte, all_lte


class Loader(object):

    def __init__(self, path):
        self._document = fitz.Document(path)

    def __enter(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._document.close()

    def lines(self, page=0, color=0.0, reverse=False):
        doc_page = self._document[page]
        for path in doc_page.get_drawings():
            pass



def fitz_(path):
    import fitz
    doc = fitz.Document(path)
    page = doc[5]
    atestttt = page.get_text("words")
    paths = page.get_drawings()  # extract existing drawings
    # this is a list of "paths", which can directly be drawn again using Shape
    # -------------------------------------------------------------------------
    #
    # define some output page with the same dimensions
    outpdf = fitz.open()
    outpage = outpdf.new_page(width=page.rect.width, height=page.rect.height)
    shape = outpage.new_shape()  # make a drawing canvas for the output page
    # --------------------------------------
    # loop through the paths and draw them
    # --------------------------------------
    bbox_list = []
    for path in paths:
        # ------------------------------------
        # draw each entry of the 'items' list
        # ------------------------------------
        for item in path["items"]:  # these are the draw commands
            if item[0] == "l":  # line
                shape.draw_line(item[1], item[2])
            elif item[0] == "re":  # rectangle
                # if path['fill'] == (1.0, 1.0, 1.0):
                #     show_bbox(page.mediabox[2], page.mediabox[3], [item[1]])
                if path['fill'] == (0.0, 0.0, 0.0):
                    print(min(item[1][2]-item[1][0], item[1][3]-item[1][1]))
                    bbox_list.append(item[1])
                shape.draw_rect(item[1])
            elif item[0] == "qu":  # quad
                shape.draw_quad(item[1])
            elif item[0] == "c":  # curve
                shape.draw_bezier(item[1], item[2], item[3], item[4])
            else:
                raise ValueError("unhandled drawing", item)
        # ------------------------------------------------------
        # all items are drawn, now apply the common properties
        # to finish the path
        # ------------------------------------------------------
        shape.finish(
            fill=path["fill"],  # fill color
            color=path["color"],  # line color
            dashes=path["dashes"],  # line dashing
            even_odd=path.get("even_odd", True),  # control color of overlaps
            closePath=path["closePath"],  # whether to connect last and first point
            lineJoin=path["lineJoin"],  # how line joins should look like
            lineCap=max(path["lineCap"]),  # how line ends should look like
            width=path["width"],  # line width
            stroke_opacity=path.get("stroke_opacity", 1),  # same value for both
            fill_opacity=path.get("fill_opacity", 1),  # opacity parameters
        )
    # all paths processed - commit the shape to its page
    shape.commit()
    outpdf.save("drawings-page.pdf")
    show_bbox(page.mediabox[2], page.mediabox[3], bbox_list)





def fitz_bbox(path):
    import fitz
    doc = fitz.open(path)
    page = doc[5]
    paths = page.get_drawings()
    bbox_list = []
    for path in paths:
        for item in path["items"]:  # these are the draw commands
            if item[0] == "re":  # rectangle
                bbox_list.append(item[1])
    show_bbox(page.mediabox[2], page.mediabox[3], bbox_list)




if __name__ == '__main__':
    fitz_('E:/Code/Mine/github/tex/test/pdf/89df2a78460636a6fa35edb53ade119b.pdf')