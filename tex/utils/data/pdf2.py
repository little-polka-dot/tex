import fitz
import numpy as np
import cv2
import math
import enum


class Color(object):

    def __init__(self, r, g, b):
        assert 0 <= r <= 1
        assert 0 <= g <= 1
        assert 0 <= b <= 1
        self._rgb = (r, g, b)

    @property
    def r(self):
        return self._rgb[0]

    @property
    def g(self):
        return self._rgb[1]

    @property
    def b(self):
        return self._rgb[2]

    def __eq__(self, other):
        return math.isclose(self.r, other.r) and math.isclose(
            self.g, other.g) and math.isclose(self.b, other.b)

    @property
    def int_8bit(self):
        return round(255 * self.r) * (2 ** 16) + round(
            255 * self.g) * (2 ** 8) + round(255 * self.b) * (2 ** 0)

    @property
    def hex_8bit(self):
        """ 0xffffff 0x000000 ... """
        return hex(self.int_8bit)

    @classmethod
    def from_hex_8bit(cls, hex_string):
        n = int(hex_string, 16)
        b = n % 256
        assert (n - b) % (2 ** 8) == 0
        n = (n - b) // (2 ** 8)
        g = n % 256
        assert (n - g) % (2 ** 8) == 0
        n = (n - g) // (2 ** 8)
        r = n % 256
        return cls(r / 255, g / 255, b / 255)


class Point(object):

    def __init__(self, x, y):
        self._position = (x, y)

    @property
    def x(self):
        return self._position[0]

    @property
    def y(self):
        return self._position[1]

    def distance_x(self, p):
        return abs(self.x - p.x)

    def distance_y(self, p):
        return abs(self.y - p.y)

    def distance(self, p):
        return (self.distance_x(p) ** 2 + self.distance_y(p) ** 2) ** 0.5


class Angle(Point):

    class Quadrant(enum.Enum):
        first = 1
        second = 2
        third = 3
        forth = 4

    def __init__(self, x, y):
        # x=cos(theta) y=sin(theta)
        assert math.isclose(x ** 2 + y ** 2, 1)
        super().__init__(x, y)

    def quadrant(self):
        if self.x > 0 and self.y > 0:
            return self.Quadrant.first
        if self.x < 0 and self.y > 0:
            return self.Quadrant.second
        if self.x < 0 and self.y < 0:
            return self.Quadrant.third
        if self.x > 0 and self.y < 0:
            return self.Quadrant.forth


class Rectangle(object):

    def __init__(self, left, top, right, bottom):
        assert right >= left and bottom >= top
        self._rect = (left, top, right, bottom)

    @property
    def left(self):
        return self._rect[0]

    @property
    def top(self):
        return self._rect[1]

    @property
    def right(self):
        return self._rect[2]

    @property
    def bottom(self):
        return self._rect[3]

    def __eq__(self, other):
        return math.isclose(self.left, other.left) and math.isclose(self.top, other.top) \
            and math.isclose(self.right, other.right) and math.isclose(self.bottom, other.bottom)

    def copy(self):
        return self.__class__(self.left, self.top, self.right, self.bottom)

    @property
    def width(self):
        return self.right - self.left

    w = W = width

    @property
    def height(self):
        return self.bottom - self.top

    h = H = height

    @property
    def center(self):
        return Point(self.center_x, self.center_y)

    @property
    def center_x(self):
        return (self.left + self.right) / 2

    @property
    def center_y(self):
        return (self.top + self.bottom) / 2

    @property
    def x(self):
        return self.left

    @property
    def y(self):
        return self.top

    def min_distance_x(self, rect):
        return max(0, abs(self.center_x - rect.center_x) - (self.w + rect.w) / 2)

    def min_distance_y(self, rect):
        return max(0, abs(self.center_y - rect.center_y) - (self.h + rect.h) / 2)

    def min_distance(self, rect):
        return (self.min_distance_x(rect) ** 2 + self.min_distance_y(rect) ** 2) ** 0.5

    def move_x(self, offset=0):
        """ X轴平移 """
        return self.__class__(
            self.left + offset, self.top, self.right + offset, self.bottom)

    def move_y(self, offset=0):
        """ Y轴平移 """
        return self.__class__(
            self.left, self.top + offset, self.right, self.bottom + offset)

    def intersect(self, rect):
        """ 交集矩形 """
        if self.min_distance_x(rect) <= 0 and self.min_distance_y(rect) <= 0:
            return self.__class__(max(self.left, rect.left), max(self.top, rect.top),
                min(self.right, rect.right), min(self.bottom, rect.bottom))

    def mbr(self, rect):
        """ 最小外接矩形 """
        return self.__class__(min(self.left, rect.left), min(self.top, rect.top),
            max(self.right, rect.right), max(self.bottom, rect.bottom))

    minimum_bounding_rectangle = MBR = mbr

    def horizontal_closer(self, rect, max_gap=0, max_distance=None):
        return abs(self.top - rect.top) <= max_gap and abs(self.bottom - rect.bottom) <= max_gap and \
            (max_distance is None or self.min_distance_x(rect) <= max_distance)

    def line_horizontal_closer(self, rect, max_gap=0, max_distance=None):
        return self.is_horizontal() and rect.is_horizontal() and self.min_distance_y(rect) <= max_gap and \
            (max_distance is None or self.min_distance_x(rect) <= max_distance)

    def vertical_closer(self, rect, max_gap=0, max_distance=None):
        return abs(self.left - rect.left) <= max_gap and abs(self.right - rect.right) <= max_gap and \
            (max_distance is None or self.min_distance_y(rect) <= max_distance)

    def line_vertical_closer(self, rect, max_gap=0, max_distance=None):
        return self.is_vertical() and rect.is_vertical() and self.min_distance_x(rect) <= max_gap and \
            (max_distance is None or self.min_distance_y(rect) <= max_distance)

    def is_horizontal(self):
        return self.width > self.height

    def is_vertical(self):
        return self.width < self.height

    def line_combine(self, rect, max_gap=0, max_distance=None):
        """ 适用于直线矩形区域之间的合并 """
        if self.line_horizontal_closer(rect, max_gap, max_distance) or \
            self.line_vertical_closer(rect, max_gap, max_distance): return self.mbr(rect)

    def combine(self, rect, max_gap=0, max_distance=None):
        """ 适用于点矩形区域之间的合并 """
        if self.horizontal_closer(rect, max_gap, max_distance) or \
            self.vertical_closer(rect, max_gap, max_distance): return self.mbr(rect)

    def line_intersect(self, rect, max_margin=0):
        """ 计算两个线的交点 """

        def try_extend_x(obj, length):
            return obj.__class__(
                obj.left - length, obj.top, obj.right + length, obj.bottom) if obj.is_horizontal() else obj

        def try_extend_y(obj, length):
            return obj.__class__(
                obj.left, obj.top - length, obj.right, obj.bottom + length) if obj.is_vertical() else obj

        if (self.is_horizontal() and rect.is_vertical()) or (self.is_vertical() and rect.is_horizontal()):
            x_dist_value, y_dist_value = self.min_distance_x(rect), self.min_distance_y(rect)
            if x_dist_value <= max_margin and y_dist_value <= max_margin:
                return try_extend_y(try_extend_x(self, x_dist_value + rect.w), y_dist_value + rect.h) \
                    .intersect(try_extend_y(try_extend_x(rect, x_dist_value + self.w), y_dist_value + self.h))


# idea:
# 从PDF中拿到line/rect/image/char等对象并挑选出合适的数据构建Rectangle
# 通过规则列表从Rectangle提取出PointRectangle与LineRectangle
# 将PointRectangle合并为LineRectangle并添加进LineRectangle集合 接着合并LineRectangle集合(处理相连的或者是重复的线条)
# 三种线条是不可见的 1) 线宽为0; 2) 线条颜色等于背景色（有可能线条只会消失一部分）; 3) 被带颜色的矩形区域覆盖（尚不确定是否不可见）

# 1 合并点线 combine
# 2 合并线条 line_combine
# 3 计算合并后的线条之间的交点 由此得到单元格区域信息





if __name__ == '__main__':
    bg = np.zeros((500, 500, 3))
    b = Rectangle(50, 10, 60, 300)
    a = Rectangle(20, 260, 200, 270)
    bg = cv2.rectangle(bg, (a.left, a.top), (a.right, a.bottom), (255, 0, 0))
    bg = cv2.rectangle(bg, (b.left, b.top), (b.right, b.bottom), (255, 0, 0))
    c = a.line_intersect(b, 5)
    if c:
        bg = cv2.rectangle(bg, (int(c.left), int(c.top)), (int(c.right), int(c.bottom)), (0, 0, 255))
    cv2.imshow('', bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # bg = np.zeros((500, 500, 3))
    # a = Rectangle(50, 50, 200, 100)
    # b = Rectangle(300, 100, 450, 150)
    # bg = cv2.rectangle(bg, (a.left, a.top), (a.right, a.bottom), (255, 0, 0))
    # bg = cv2.rectangle(bg, (b.left, b.top), (b.right, b.bottom), (255, 0, 0))
    # c = a.line_combine(b, max_distance=100)
    # if c:
    #     bg = cv2.rectangle(bg, (int(c.left), int(c.top)), (int(c.right), int(c.bottom)), (0, 0, 255))
    # cv2.imshow('', bg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()