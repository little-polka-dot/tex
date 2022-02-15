import torch

from tex.datasets.labels import StructLang
import random
import numpy as np
import cv2
import math


class ConStructureTransform(object):

    threshold_type = cv2.THRESH_TOZERO  # 超出阈值的保持不变 低于阈值的取0

    @staticmethod
    def square_padding(image, padding=0):
        """ 矩形图片填充为正方形 """
        return cv2.copyMakeBorder(
            image,
            int((max(image.shape[:2]) - image.shape[0]) / 2),
            int((max(image.shape[:2]) - image.shape[0] + 1) / 2),
            int((max(image.shape[:2]) - image.shape[1]) / 2),
            int((max(image.shape[:2]) - image.shape[1] + 1) / 2),
            cv2.BORDER_CONSTANT, value=padding
        )

    @staticmethod
    def gaussian_noise(image, loc=0, scale=0.01):
        """ loc: 平均值 scale: 标准差 """
        return getattr(np, str(image.dtype))(np.clip(
            image / 255 + np.random.normal(loc, scale, image.shape), 0, 1) * 255)

    def __init__(self, image_size, seq_len, normalize_position=True,
                 gaussian_noise=None, film_mode=False, gaussian_blur=None, threshold=None):
        self._image_size = image_size
        self._seq_len = seq_len
        self._normalize_position = normalize_position
        self._gaussian_noise = gaussian_noise
        self._film_mode = film_mode
        self._gaussian_blur = gaussian_blur
        self._threshold = threshold

    def __call__(self, x_data, y_data):
        # TODO 图片的随机resize
        description = StructLang.from_object(y_data['description'])
        seq_labels = np.array(description.labels(self._seq_len, False, True))
        seq_inputs = np.array(description.labels(self._seq_len,  True, True))

        if self._normalize_position:  # 坐标归一化
            # TODO: 坐标归一化逻辑有问题 参考PosStructureTransform
            seq_positions = np.array([
                [
                    i[0] / x_data.shape[1],  # x / W
                    i[1] / x_data.shape[0],  # y / H
                    i[2] / x_data.shape[1],  # w / W
                    i[3] / x_data.shape[0],  # h / H
                ] for i in y_data['position']
            ])
        else:
            seq_positions = np.array(y_data['position'])
        seq_positions = np.vstack(  # 沿着seq_len方向拼接到指定长度
            (seq_positions, np.zeros((self._seq_len - seq_positions.shape[0], 4))))

        if self._gaussian_noise:  # 高斯噪音
            x_data = self.gaussian_noise(
                x_data, self._gaussian_noise['loc'], self._gaussian_noise['scale'])

        if not self._film_mode:  # 颜色反转
            x_data = np.ones_like(x_data, np.uint8) * 255 - x_data

        if self._threshold:  # 图像二值化
            _, x_data = cv2.threshold(x_data, int(self._threshold * 255), 255, self.threshold_type)

        if self._gaussian_blur:  # 高斯模糊
            for cfg_item in self._gaussian_blur:
                x_data = cv2.GaussianBlur(x_data, [cfg_item['kernel']] * 2, cfg_item['sigma'])

        x_data = cv2.resize(  # resize
            x_data, (
                int((self._image_size / max(x_data.shape[:2])) * x_data.shape[1]),
                int((self._image_size / max(x_data.shape[:2])) * x_data.shape[0]),
            )
        )

        x_data = self.square_padding(x_data / 255, padding=0)  # 正方形填充 / 归一化

        return (x_data, seq_inputs), (seq_labels, seq_positions)


class PosStructureTransform(object):

    label_alignment = (StructLang.Vocab.CELL.value, StructLang.Vocab.HEAD.value)

    def __init__(self, enc_len, dec_len):
        self._enc_len = enc_len
        self._dec_len = dec_len

    def __call__(self, x_data, y_data):  # 坐标格式必须为x-y-w-h
        # x_data [enc_len, 4] y_data description-object, [dec_len, 4]

        y_description, y_position = StructLang.from_object(y_data[0]), y_data[1]
        x_data, y_position = np.unique(np.array(x_data), axis=0), np.array(y_position)

        # 表结构描述的具有边界的单元格数一定与边界集合数目相同
        assert y_position.shape[0] == len(
            [i for i in sum(y_description.data, []) if i.value in self.label_alignment])

        # 计算线条与文本框集合的最小外接矩形
        # 有线的情况下线的最小外接矩形作为边界 否则单元格最小外接矩形会作为边界
        boundary_x = min(x_data[:, 0].min(), y_position[:, 0].min())
        boundary_y = min(x_data[:, 1].min(), y_position[:, 1].min())
        boundary_x_max = max(
            (x_data[:, 0] + x_data[:, 2]).max(), (y_position[:, 0] + y_position[:, 2]).max())
        boundary_y_max = max(
            (x_data[:, 1] + x_data[:, 3]).max(), (y_position[:, 1] + y_position[:, 3]).max())
        boundary_w = boundary_x_max - boundary_x
        boundary_h = boundary_y_max - boundary_y

        # decoder输入与输出中的表结构描述 输入比输出多一个开始占位符
        seq_labels = np.array(y_description.labels(self._dec_len, False, True))
        seq_inputs = np.array(y_description.labels(self._dec_len,  True, True))

        # 坐标归一化 是否可以认为归一化可以去除掉尺度以及绝对坐标的信息？
        normalize_size = max(boundary_w, boundary_h)
        normalize_fill = abs(boundary_w - boundary_h) / 2
        if boundary_w > boundary_h:
            x_data = np.array([
                [
                    (i[0] - boundary_x) / normalize_size,  # x / W
                    (i[1] - boundary_y + normalize_fill) / normalize_size,  # y / H
                    i[2] / normalize_size,  # w / W
                    i[3] / normalize_size,  # h / H
                ] for i in x_data
            ])
            seq_positions = np.array([
                [
                    (i[0] - boundary_x) / normalize_size,  # x / W
                    (i[1] - boundary_y + normalize_fill) / normalize_size,  # y / H
                    i[2] / normalize_size,  # w / W
                    i[3] / normalize_size,  # h / H
                ] for i in y_position
            ])
        else:
            x_data = np.array([
                [
                    (i[0] - boundary_x + normalize_fill) / normalize_size,  # x / W
                    (i[1] - boundary_y) / normalize_size,  # y / H
                    i[2] / normalize_size,  # w / W
                    i[3] / normalize_size,  # h / H
                ] for i in x_data
            ])
            seq_positions = np.array([
                [
                    (i[0] - boundary_x + normalize_fill) / normalize_size,  # x / W
                    (i[1] - boundary_y) / normalize_size,  # y / H
                    i[2] / normalize_size,  # w / W
                    i[3] / normalize_size,  # h / H
                ] for i in y_position
            ])

        # (Xn, Yn, Wn, Hn) -> (Wn, Hn, Xn-X1, Yn-Y1, Xn-X2, Yn-Y2, ..., Xn-Xn, Yn-Yn)
        # l_value = np.tile(x_data[:, :2], (1, x_data.shape[0]))
        # r_value = np.tile(x_data[:, :2].flatten(), (x_data.shape[0], 1))
        # x_data = np.hstack((x_data[:, 2:], l_value - r_value))

        # 线条与文本框坐标信息集合结尾处填充(0,0,0,0)
        x_data = np.vstack(
            (x_data, np.zeros((self._enc_len - x_data.shape[0], x_data.shape[1]))))

        # 对齐单元格坐标信息与表结构描述 对seq_positions按照y-x排序 然后根据seq_labels填充(0,0,0,0)
        seq_positions = seq_positions[np.lexsort((seq_positions[:, 0], seq_positions[:, 1])), :]
        for label_index, label in enumerate(seq_labels):
            # 仅有cell与head两个类型的单元格有坐标描述信息
            if label not in self.label_alignment:
                seq_positions = np.insert(
                    seq_positions, label_index, values=np.array([0, 0, 0, 0]), axis=0)

        # 确认序列长度是否一致
        assert seq_inputs.shape[0] == seq_labels.shape[0] == seq_positions.shape[0] == self._dec_len
        assert x_data.shape[0] == self._enc_len

        return (x_data, seq_inputs), (seq_labels, seq_positions)


class ResStructureTransform(object):

    label_alignment = (StructLang.Vocab.CELL.value, StructLang.Vocab.HEAD.value)

    def __init__(self, enc_len, dec_len):
        self._enc_len = enc_len
        self._dec_len = dec_len

    def rectangle_mask(self, original, rect, blur=False):
        x_min = int(rect[0] * self._enc_len)
        x_len = int(rect[2] * self._enc_len)
        x_max = x_min + x_len
        y_min = int(rect[1] * self._enc_len)
        y_len = int(rect[3] * self._enc_len)
        y_max = y_min + y_len
        if blur:
            def mask(p): return math.cos(math.pi * p / 2)
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    x_v = mask(2 * abs(x - x_center) / x_len)
                    y_v = mask(2 * abs(y - y_center) / y_len)
                    original[x, y] = min(x_v, y_v)
        else:
            original[x_min: x_max + 1, y_min: y_max + 1] = 1
        return original

    def __call__(self, x_data, y_data):  # 坐标格式必须为x-y-w-h

        # x_data [enc_len, 4], [enc_len, 4] y_data description-object, [dec_len, 4]

        line_data, text_data = map(lambda i: np.unique(np.array(i), axis=0), x_data)

        y_description, y_position = StructLang.from_object(y_data[0]), np.array(y_data[1])

        # 表结构描述的具有边界的单元格数一定与边界集合数目相同
        assert y_position.shape[0] == len(
            [i for i in sum(y_description.data, []) if i.value in self.label_alignment])

        # 计算线条与文本框集合的最小外接矩形
        # 有线的情况下线的最小外接矩形作为边界 否则单元格最小外接矩形会作为边界
        boundary_x = min(line_data[:, 0].min(), text_data[:, 0].min(), y_position[:, 0].min())
        boundary_y = min(line_data[:, 1].min(), text_data[:, 1].min(), y_position[:, 1].min())
        boundary_x_max = max((line_data[:, 0] + line_data[:, 2]).max(),
            (text_data[:, 0] + text_data[:, 2]).max(), (y_position[:, 0] + y_position[:, 2]).max())
        boundary_y_max = max((line_data[:, 1] + line_data[:, 3]).max(),
            (text_data[:, 1] + text_data[:, 3]).max(), (y_position[:, 1] + y_position[:, 3]).max())
        boundary_w = boundary_x_max - boundary_x
        boundary_h = boundary_y_max - boundary_y

        # decoder输入与输出中的表结构描述 输入比输出多一个开始占位符
        seq_labels = np.array(y_description.labels(self._dec_len, False, True))
        seq_inputs = np.array(y_description.labels(self._dec_len,  True, True))

        # 坐标归一化 是否可以认为归一化可以去除掉尺度以及绝对坐标的信息？
        normalize_size = max(boundary_w, boundary_h)
        normalize_fill = abs(boundary_w - boundary_h) / 2
        if boundary_w > boundary_h:
            line_data = np.array([
                [
                    (i[0] - boundary_x) / normalize_size,  # x / W
                    (i[1] - boundary_y + normalize_fill) / normalize_size,  # y / H
                    i[2] / normalize_size,  # w / W
                    i[3] / normalize_size,  # h / H
                ] for i in line_data
            ])
            text_data = np.array([
                [
                    (i[0] - boundary_x) / normalize_size,  # x / W
                    (i[1] - boundary_y + normalize_fill) / normalize_size,  # y / H
                    i[2] / normalize_size,  # w / W
                    i[3] / normalize_size,  # h / H
                ] for i in text_data
            ])
            seq_positions = np.array([
                [
                    (i[0] - boundary_x) / normalize_size,  # x / W
                    (i[1] - boundary_y + normalize_fill) / normalize_size,  # y / H
                    i[2] / normalize_size,  # w / W
                    i[3] / normalize_size,  # h / H
                ] for i in y_position
            ])
        else:
            line_data = np.array([
                [
                    (i[0] - boundary_x + normalize_fill) / normalize_size,  # x / W
                    (i[1] - boundary_y) / normalize_size,  # y / H
                    i[2] / normalize_size,  # w / W
                    i[3] / normalize_size,  # h / H
                ] for i in line_data
            ])
            text_data = np.array([
                [
                    (i[0] - boundary_x + normalize_fill) / normalize_size,  # x / W
                    (i[1] - boundary_y) / normalize_size,  # y / H
                    i[2] / normalize_size,  # w / W
                    i[3] / normalize_size,  # h / H
                ] for i in text_data
            ])
            seq_positions = np.array([
                [
                    (i[0] - boundary_x + normalize_fill) / normalize_size,  # x / W
                    (i[1] - boundary_y) / normalize_size,  # y / H
                    i[2] / normalize_size,  # w / W
                    i[3] / normalize_size,  # h / H
                ] for i in y_position
            ])

        # 线条与文本框坐标信息集合转为位图
        x_data = np.zeros((2, self._enc_len, self._enc_len))  # [channel, x, y]
        for rect in line_data: x_data[0] = self.rectangle_mask(x_data[0], rect)
        for rect in text_data: x_data[1] = self.rectangle_mask(x_data[1], rect, True)

        # 对齐单元格坐标信息与表结构描述 对seq_positions按照y-x排序 然后根据seq_labels填充(0,0,0,0)
        seq_positions = seq_positions[np.lexsort((seq_positions[:, 0], seq_positions[:, 1])), :]
        for label_index, label in enumerate(seq_labels):
            # 仅有cell与head两个类型的单元格有坐标描述信息
            if label not in self.label_alignment:
                seq_positions = np.insert(
                    seq_positions, label_index, values=np.array([0, 0, 0, 0]), axis=0)

        # 确认序列长度是否一致
        assert seq_inputs.shape[0] == seq_labels.shape[0] == seq_positions.shape[0] == self._dec_len

        return (x_data, seq_inputs), (seq_labels, seq_positions)


if __name__ == '__main__':
    transform = ResStructureTransform(224, 512)
    with open(r'E:\Code\Mine\github\tex\test\pdf\data.json', 'r', encoding='utf-8') as jf:
        import json
        data = json.load(jf)
        for item in data:
            print(item['DESC'])
            x = item['X']
            y = item['Y']['description'], item['Y']['position']
            x_, y_ = transform(x, y)
            b = 800
            g_0 = x_[0][0].T
            g_1 = x_[0][1].T
            g_2 = np.zeros((b, b))
            # print('单元格')
            for idx, i in enumerate(y_[1]):  # 单元格
                i = [int(r * b) for r in i]
                m = ['PAD','SOS','EOS','ENTER','CELL','HEAD','TAIL_H','TAIL_V','TAIL_T']
                # print(m[y_[0][idx]], i)
                g_2 = cv2.rectangle(
                    g_2, (i[0], i[1]), (i[0] + i[2], i[1] + i[3]), 1)
                # if i[0] > 0 or i[1] > 0 or i[2] > 0 or i[3] > 0:
                #     cv2.imshow('2', g_2)
                #     print(i)
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()
            cv2.imshow('0', g_0)
            cv2.imshow('1', g_1)
            # cv2.imshow('2', g_2)
            cv2.imshow('mask.png', cv2.max(g_0, g_1))
            cv2.waitKey(0)
            cv2.destroyAllWindows()


