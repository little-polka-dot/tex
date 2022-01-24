import torch

from tex.datasets.labels import StructLang
import random
import numpy as np
import cv2


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
                 gaussian_noise=None, flim_mode=False, gaussian_blur=None, threshold=None):
        self._image_size = image_size
        self._seq_len = seq_len
        self._normalize_position = normalize_position
        self._gaussian_noise = gaussian_noise
        self._flim_mode = flim_mode
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

        if not self._flim_mode:  # 颜色反转
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

    def __init__(self, enc_len, dec_len, normalize_position=True, transform_position=False):
        self._enc_len = enc_len
        self._dec_len = dec_len
        self._normalize_position = normalize_position
        self._transform_position = transform_position

    def __call__(self, x_data, y_data):
        # x_data [enc_len, 4] y_data {description:StructLang position:[dec_len, 4]}

        # description的条数需要大于等于position条数
        assert y_data['description'].size >= y_data['position'].shape[0]

        # y_data['position']一定在x_data集合最小外接矩形的范围内
        assert (x_data[:, 0] + x_data[:, 2]).max() >= (
                y_data['position'][:, 0] + y_data['position'][:, 2]).max()
        assert (x_data[:, 1] + x_data[:, 3]).max() >= (
                y_data['position'][:, 1] + y_data['position'][:, 3]).max()
        assert x_data[:, 0].min() <= y_data['position'][:, 0].min()
        assert x_data[:, 1].min() <= y_data['position'][:, 1].min()

        boundary_w = (x_data[:, 0] + x_data[:, 2]).max() - x_data[:, 0].min()
        boundary_h = (x_data[:, 1] + x_data[:, 3]).max() - x_data[:, 1].min()

        seq_labels = np.array(y_data['description'].labels(self._dec_len, False, True))
        seq_inputs = np.array(y_data['description'].labels(self._dec_len,  True, True))

        if self._normalize_position:  # 坐标归一化
            # 是否可以认为归一化可以去除掉尺度以及绝对坐标的信息？
            normalize_size = max(boundary_w, boundary_h)
            normalize_fill = abs(boundary_w - boundary_h) / 2
            if boundary_w > boundary_h:
                x_data = np.array([
                    [
                        i[0] / normalize_size,  # x / W
                        (i[1] + normalize_fill) / normalize_size,  # y / H
                        i[2] / normalize_size,  # w / W
                        i[3] / normalize_size,  # h / H
                    ] for i in x_data
                ])
                seq_positions = np.array([
                    [
                        i[0] / normalize_size,  # x / W
                        (i[1] + normalize_fill) / normalize_size,  # y / H
                        i[2] / normalize_size,  # w / W
                        i[3] / normalize_size,  # h / H
                    ] for i in y_data['position']
                ])
            else:
                x_data = np.array([
                    [
                        (i[0] + normalize_fill) / normalize_size,  # x / W
                        i[1] / normalize_size,  # y / H
                        i[2] / normalize_size,  # w / W
                        i[3] / normalize_size,  # h / H
                    ] for i in x_data
                ])
                seq_positions = np.array([
                    [
                        (i[0] + normalize_fill) / normalize_size,  # x / W
                        i[1] / normalize_size,  # y / H
                        i[2] / normalize_size,  # w / W
                        i[3] / normalize_size,  # h / H
                    ] for i in y_data['position']
                ])
        else:
            x_data, seq_positions = np.array(x_data), np.array(y_data['position'])

        # 将seq_positions与seq_labels对齐 对seq_positions按照y-x排序 然后根据seq_labels填充(0,0,0,0)
        seq_positions = seq_positions[np.lexsort((seq_positions[:, 0], seq_positions[:, 1])), :]
        for label_index, label in enumerate(seq_labels):
            if label not in (StructLang.Vocab.CELL.value, StructLang.Vocab.HEAD.value):
                seq_positions = np.insert(seq_positions, label_index, values=np.array([0, 0, 0, 0]), axis=0)

        if self._transform_position:  # TODO: [测试] x_data高维空间映射
            # (Xn, Yn, Wn, Hn) -> (Wn, Hn, Xn-X1, Yn-Y1, Xn-X2, Yn-Y2, ..., Xn-Xn, Yn-Yn)
            # TODO: 打乱序列顺序后模型是否还能正常识别？
            l_value = np.tile(x_data[:, :2], (1, x_data.shape[0]))
            r_value = np.tile(x_data[:, :2].flatten(), (x_data.shape[0], 1))
            x_data = np.hstack((x_data[:, 2:], l_value - r_value))

        x_data = np.vstack((x_data, np.zeros((self._enc_len - x_data.shape[0], x_data.shape[1]))))
        seq_positions = np.vstack((seq_positions, np.zeros(
            (self._dec_len - seq_positions.shape[0], seq_positions.shape[1]))))

        return (x_data, seq_inputs), (seq_labels, seq_positions)


if __name__ == '__main__':
    transform = PosStructureTransform(200, 200)
    with open('E:/Code/Mine/github/tex/test/pdf/ea9858d5486bd46e04ede04a9a69db6.json', 'r', encoding='utf-8') as jf:
        import json
        data = json.load(jf)
        for item in data[1:]:
            transform(np.array(item['X']), {'description': StructLang.from_object(item['Y']['description']), 'position': np.array(item['Y']['position'])})
            input()


