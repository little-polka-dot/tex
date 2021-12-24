import numpy as np
import torch
import torch.nn as nn
from tex.models.structure.encoder import resnet
from tex.models.structure.encoder import gc


class Encoder(nn.Module):

    def __init__(self, d_input, d_model, block, layers):
        super(Encoder, self).__init__()
        self.pre_process = nn.Sequential(
            nn.Conv2d(d_input, 64, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(128, 128, (1, 1), bias=False),
        )
        if isinstance(block, str): block = getattr(resnet, block)
        self.layers = nn.ModuleList([
            resnet.make_layer(
                layers[0],
                block,
                128,
                128,
                stride=(1, 1),
                sub=gc.GlobalContextBlock(
                    128 * block.expansion, 128 * block.expansion)
            ),
            resnet.make_layer(
                layers[1],
                block,
                128 * block.expansion,
                256,
                stride=(2, 2),
                sub=gc.GlobalContextBlock(
                    256 * block.expansion, 256 * block.expansion)
            ),
            resnet.make_layer(
                layers[2],
                block,
                256 * block.expansion,
                512,
                stride=(2, 2),
                sub=gc.GlobalContextBlock(
                    512 * block.expansion, 512 * block.expansion)
            ),
            resnet.make_layer(
                layers[3],
                block,
                512 * block.expansion,
                1024,
                stride=(2, 2),
                sub=gc.GlobalContextBlock(
                    1024 * block.expansion, 1024 * block.expansion)
            ),
        ])
        self.layers_map = nn.ModuleList([
            nn.Conv2d(
                256 * block.expansion, d_model, (1, 1), bias=False),
            nn.Conv2d(
                512 * block.expansion, d_model, (1, 1), bias=False),
            nn.Conv2d(
                1024 * block.expansion, d_model, (1, 1), bias=False)
        ])

    def forward(self, x):
        """
        将细粒度，中粒度，粗粒度三个特征图合并为一个向量
        输出： [batch_size, img_len(取决于输入图像的size), d_model]
        """
        x = self.pre_process(x)
        output = None
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx > 0:
                vec = self.layers_map[idx-1](x)
                vec = vec.view(vec.size(0), vec.size(1), -1)
                if output is None:
                    output = vec
                else:
                    output = torch.cat((output, vec), -1)
        return output.transpose(1, 2)


if __name__ == '__main__':
    net = Encoder(1, 256, resnet.BasicBlock, [3, 4, 6, 3])
    i = torch.tensor(np.random.random((10, 1, 224, 224)), dtype=torch.float)
    print(net(i).size())
