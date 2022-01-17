from tex.datasets.labels import StructLang
import random
import numpy as np
import cv2


class BackboneStructureTransformer(object):

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
        cut_len = random.randint(1, description.size + 1)  # 随机截断表格描述语句(用于模型训练)
        # TODO 无须随机截断 对于transformer来说没有意义
        seq_labels = np.array(description.labels(self._seq_len, False, True, cut_len))
        seq_inputs = np.array(description.labels(self._seq_len, True, False, cut_len))

        if self._normalize_position:  # 坐标归一化
            seq_position = np.array([
                [
                    i[0] / x_data.shape[1],  # x / W
                    i[1] / x_data.shape[0],  # y / H
                    i[2] / x_data.shape[1],  # w / W
                    i[3] / x_data.shape[0],  # h / H
                ] for i in y_data['position']
            ])
        else:
            seq_position = np.array(y_data['position'])

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

        return (x_data, seq_inputs), (seq_labels, seq_position)


if __name__ == '__main__':
    x = cv2.imread('F:/img.png', cv2.IMREAD_GRAYSCALE)
    print(x.shape)

    st = StructLang(2, 6)
    st.merge_cell((0, 1), (0, 2))
    st.merge_cell((0, 3), (1, 3))
    st.merge_cell((0, 4), (1, 5))
    st.merge_cell((1, 0), (1, 1))

    y = {
        'description': st.to_object(),
        'position': [[0, 0, 0, 0], [12, 12, 500, 32]],
    }

    (xd, si), (sl, sp) = BackboneStructureTransformer(**{
        'seq_len': 256,
        'image_size': 800,
        'normalize_position': True,
        'gaussian_noise': False,
        'flim_mode': False,
        'gaussian_blur': None,
        'threshold': 0.1,
    })(x, y)
    print(xd.shape)
    cv2.imshow('img', xd)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # print(x_data.shape)
    # print(seq_inputs)
    # print(seq_labels)
    # print(seq_position)
