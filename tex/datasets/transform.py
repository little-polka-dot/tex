from tex.datasets.labels import StructLang
import random
import numpy as np
import cv2


class BackboneStructureTransform(object):

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
            # TODO: 坐标归一化逻辑有问题 参考PositionalStructureTransform
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


class PositionalStructureTransform(object):

    def __init__(self, enc_len, dec_len, normalize_position=True):
        self._enc_len = enc_len
        self._dec_len = dec_len
        self._normalize_position = normalize_position

    def __call__(self, x_data, y_data):
        # x_data [enc_len, 4] y_data description:[dec_len] position:[dec_len, 4]
        # TODO: 检测y_data['position']一定在x_data范围内
        boundary_w = (x_data[:, 0] + x_data[:, 2]).max() - (x_data[:, 0].min())
        boundary_h = (x_data[:, 1] + x_data[:, 3]).max() - (x_data[:, 1].min())
        description = StructLang.from_object(y_data['description'])
        seq_labels = np.array(description.labels(self._dec_len, False, True))
        seq_inputs = np.array(description.labels(self._dec_len,  True, True))

        if boundary_w > boundary_h:  # 边界填充计算
            x_data[:, 1] = x_data[:, 1] + (boundary_w - boundary_h) / 2
            y_data['position'][:, 1] = y_data['position'][:, 1] + (boundary_w - boundary_h) / 2
        else:
            x_data[:, 0] = x_data[:, 0] + (boundary_h - boundary_w) / 2
            y_data['position'][:, 0] = y_data['position'][:, 0] + (boundary_h - boundary_w) / 2

        if self._normalize_position:  # 坐标归一化
            x_data = np.array([
                [
                    i[0] / W,  # x / W
                    i[1] / H,  # y / H
                    i[2] / W,  # w / W
                    i[3] / H,  # h / H
                ] for i in x_data
            ])
            seq_positions = np.array([
                [
                    i[0] / W,  # x / W
                    i[1] / H,  # y / H
                    i[2] / W,  # w / W
                    i[3] / H,  # h / H
                ] for i in y_data['position']
            ])
        else:
            x_data = np.array(x_data)
            seq_positions = np.array(y_data['position'])

        seq_positions = np.vstack(  # 沿着seq_len方向拼接到指定长度
            (seq_positions, np.zeros((self._dec_len - seq_positions.shape[0], 4))))


if __name__ == '__main__':
    # x = cv2.imread('F:/img.png', cv2.IMREAD_GRAYSCALE)
    # print(x.shape)
    #
    # st = StructLang(2, 6)
    # st.merge_cell((0, 1), (0, 2))
    # st.merge_cell((0, 3), (1, 3))
    # st.merge_cell((0, 4), (1, 5))
    # st.merge_cell((1, 0), (1, 1))
    #
    # y = {
    #     'description': st.to_object(),
    #     'position': [[0, 0, 0, 0], [12, 12, 500, 32]],
    # }
    #
    # (xd, si), (sl, sp) = BackboneStructureTransform(**{
    #     'seq_len': 256,
    #     'image_size': 800,
    #     'normalize_position': True,
    #     'gaussian_noise': False,
    #     'flim_mode': False,
    #     'gaussian_blur': None,
    #     'threshold': 0.1,
    # })(x, y)
    # print(xd.shape)
    # cv2.imshow('img', xd)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # print(x_data.shape)
    # # print(seq_inputs)
    # # print(seq_labels)
    # # print(seq_position)

    x = np.random.random((6, 4))
    print(x)
    print((x[:, 0] < x[:, 1]).max())