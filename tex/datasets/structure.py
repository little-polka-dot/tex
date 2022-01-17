from torch.utils.data import Dataset
import os
import cv2
import json


class BackboneStructureDataset(Dataset):

    imread_scale = cv2.IMREAD_GRAYSCALE

    def __init__(self, path='.data', transform=None):
        """
        文件目录结构如下：
            - path
                - X
                    - *.png
                - Y
                    - *.txt
                - INDEX
        """
        self.transform = transform
        assert os.path.exists(path)
        self.i = os.path.join(path, 'INDEX')
        assert os.path.exists(self.i)
        self.x = os.path.join(path, 'X')
        assert os.path.exists(self.x)
        self.y = os.path.join(path, 'Y')
        assert os.path.exists(self.y)
        with open(self.i, 'r') as index_file:
            self.index = [
                i.strip() for i in index_file if i]

    def __len__(self): return len(self.index)

    def __getitem__(self, index):
        x_path = os.path.join(
            self.x, self.index[index] + '.png')
        assert os.path.exists(x_path)
        y_path = os.path.join(
            self.y, self.index[index] + '.txt')
        assert os.path.exists(y_path)
        x_data = cv2.imread(
            x_path, self.imread_scale)
        with open(y_path, 'r') as y_file:
            y_data = json.load(y_file)
        if callable(self.transform):
            x_data, y_data = \
                self.transform(x_data, y_data)
        return x_data, y_data

    def add(self, index_name, x_data, y_data):
        assert index_name not in self.index
        x_path = os.path.join(
            self.x, index_name + '.png')
        assert not os.path.exists(x_path)
        y_path = os.path.join(
            self.y, index_name + '.txt')
        assert not os.path.exists(y_path)
        cv2.imwrite(x_path, x_data)
        with open(y_path, 'w') as y_file:
            json.dump(y_data, y_file)
        with open(self.i, 'a') as index_file:
            index_file.write(f'{index_name}\n')
        self.index.append(index_name)


if __name__ == '__main__':
    p = 'E:/Code/Mine/github/tex/test/.data/structure/train'
    config = {
        'seq_len': 256,
        'image_size': 227,
        'normalize_position': True,
        'gaussian_noise': False,
        'flim_mode': False,
        'gaussian_blur': [
            {
                'kernel': 3,
                'sigma': 0,
            },
        ],
        'threshold': None,
    }
    sd = BackboneStructureDataset(p, transform=config)

    # st = StructLang(2, 6)
    # st.merge_cell((0, 1), (0, 2))
    # st.merge_cell((0, 3), (1, 3))
    # st.merge_cell((0, 4), (1, 5))
    # st.merge_cell((1, 0), (1, 1))
    #
    # yd = {
    #     'description': st.to_object(),
    #     'position': [[0, 0, 0, 0], [100, 100, 100, 100]]
    # }
    #
    # im = cv2.imread('F:/img.png')
    #
    # sd.add('1', im, yd)
    # sd.add('2', im, yd)
    # sd.add('3', im, yd)
    # sd.add('4', im, yd)
    # sd.add('5', im, yd)

    import torch

    dl = torch.utils.data.DataLoader(sd, batch_size=3)

    for _ in range(10):
        for x, y in dl:
            print(x[0].size())
            print(x[1].size())
            print(y[0].size())
            print(y[1].size())

