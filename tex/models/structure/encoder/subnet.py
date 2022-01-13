import torch
import torch.nn as nn


class ContextModeling(nn.Module):

    def __init__(self, channels):
        super(ContextModeling, self).__init__()
        self.score_conv = nn.Conv2d(channels, 1, (1, 1))

    def forward(self, x):  # x: [batch_size, c, h, w]
        score = self.score_conv(x).view(x.size(0), 1, -1)  # [batch_size, 1, h*w]
        score = torch.softmax(score, dim=-1).unsqueeze(-1)  # [batch_size, 1, h*w, 1]
        x = x.view(x.size(0), x.size(1), -1).unsqueeze(1)  # [batch_size, 1, c, h*w]
        return torch.matmul(x, score).transpose(1, 2)  # [batch_size, c, 1, 1]


class Transformer(nn.Module):

    def __init__(self, channels, d_hidden):
        super(Transformer, self).__init__()
        self.conv_1 = nn.Conv2d(channels, d_hidden, (1, 1))
        self.layer_norm = nn.LayerNorm(d_hidden)
        self.conv_2 = nn.Conv2d(d_hidden, channels, (1, 1))

    def forward(self, x):
        return self.conv_2(  # [batch_size, channels, 1, 1]
            torch.relu(  # [batch_size, d_hidden, 1, 1]
                self.layer_norm(
                    self.conv_1(x).view(x.size(0), -1))  # [batch_size, d_hidden]
            ).unsqueeze(-1).unsqueeze(-1)
        )


class GlobalContextBlock(nn.Module):

    def __init__(self, channels, d_hidden):
        super(GlobalContextBlock, self).__init__()
        self.net = nn.Sequential(
            ContextModeling(channels), Transformer(channels, d_hidden))

    def forward(self, x): return x + self.net(x)
