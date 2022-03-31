import torch.nn as nn
import torch.nn.functional as F
import torch

def init_weights(m):
    if type(m) == nn.Conv1d or type(m) == nn.Linear:  # This initializes all layers that we have at the start.
        nn.init.xavier_uniform_(m.weight.data)

class M5(nn.Module):  # this is m5 architecture
    def __init__(self):
        super(M5, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, 4, 2)  # (in, out, filter size, stride)
        self.bn1 = nn.BatchNorm1d(128)  # normalize
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(128, 128, 2)  # by default,the stride is 1 if it is not specified here.
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(128, 256, 2)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(4)

        self.conv4 = nn.Conv1d(256, 512, 2)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(2)

        self.avgPool = nn.AdaptiveAvgPool1d(
            1)  # insteads of using nn.AvgPool1d(30) (where I need to manually check the dimension that comes in). I use adaptive n flatten
        # the advantage of adaptiveavgpool is that it manually adjust to avoid dimension issues
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 5)  # this is the output layer.

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x_4 = self.pool4(x)
        # print(x.shape)#[8, 512, 5000]
        x = self.avgPool(x_4)
        x = self.flatten(x)  # replaces permute(0,2,1) with flatten
        x = self.fc1(x)  # output layer ([n,1, 10] i.e 10 probs. for each audio files)
        # print(x.shape)#[8, 5]
        return x  # we didnt use softmax here becuz we already have that in cross entropy

class M18(nn.Module):  # this is m18 architecture
    def __init__(self):
        super(M18, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 4, 2)  # (in, out, filter size, stride)
        self.bn1 = nn.BatchNorm1d(64)  # this is used to normalize.
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(64, 64, 2)  # by default, the stride is 1 if it is not specified here.
        self.bn2 = nn.BatchNorm1d(64)
        self.conv2b = nn.Conv1d(64, 64, 2)  # by default, the stride is 1 if it is not specified here.
        self.bn2b = nn.BatchNorm1d(64)
        self.conv2c = nn.Conv1d(64, 64, 2)
        self.bn2c = nn.BatchNorm1d(64)
        self.conv2d = nn.Conv1d(64, 64, 2)
        self.bn2d = nn.BatchNorm1d(64)

        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, 2)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv3b = nn.Conv1d(128, 128, 2)
        self.bn3b = nn.BatchNorm1d(128)
        self.conv3c = nn.Conv1d(128, 128, 2)
        self.bn3c = nn.BatchNorm1d(128)
        self.conv3d = nn.Conv1d(128, 128, 2)
        self.bn3d = nn.BatchNorm1d(128)

        self.pool3 = nn.MaxPool1d(2)
        self.conv4 = nn.Conv1d(128, 256, 2)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv4b = nn.Conv1d(256, 256, 2)
        self.bn4b = nn.BatchNorm1d(256)
        self.conv4c = nn.Conv1d(256, 256, 2)
        self.bn4c = nn.BatchNorm1d(256)
        self.conv4d = nn.Conv1d(256, 256, 2)
        self.bn4d = nn.BatchNorm1d(256)

        self.pool4 = nn.MaxPool1d(2)
        self.conv5 = nn.Conv1d(256, 512, 2)
        self.bn5 = nn.BatchNorm1d(512)
        self.conv5b = nn.Conv1d(512, 512, 2)
        self.bn5b = nn.BatchNorm1d(512)
        self.conv5c = nn.Conv1d(512, 512, 2)
        self.bn5c = nn.BatchNorm1d(512)
        self.conv5d = nn.Conv1d(512, 512, 2)
        self.bn5d = nn.BatchNorm1d(512)

        self.avgPool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 5)  # this is the output layer.

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.conv2b(x)
        x = F.relu(self.bn2b(x))
        x = self.conv2c(x)
        x = F.relu(self.bn2c(x))
        x = self.conv2d(x)
        x = F.relu(self.bn2d(x))

        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv3b(x)
        x = F.relu(self.bn3b(x))
        x = self.conv3c(x)
        x = F.relu(self.bn3c(x))
        x = self.conv3d(x)
        x = F.relu(self.bn3d(x))

        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.conv4b(x)
        x = F.relu(self.bn4b(x))
        x = self.conv4c(x)
        x = F.relu(self.bn4c(x))
        x = self.conv4d(x)
        x = F.relu(self.bn4d(x))

        x = self.pool4(x)
        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = self.conv5b(x)
        x = F.relu(self.bn5b(x))
        x = self.conv5c(x)
        x = F.relu(self.bn5c(x))
        x = self.conv5d(x)
        x = F.relu(self.bn5d(x))

        x = self.avgPool(x)
        x = self.flatten(x)
        x = self.fc1(x)  # this is the output layer. [n,1, 10] i.e 10 probs for each audio files
        return x

class ResBlock(torch.nn.Module):
    def __init__(self, prev_channel, channel, conv_kernel, conv_stride, conv_pad):
        super(ResBlock, self).__init__()
        self.res = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels = prev_channel, out_channels = channel, kernel_size = conv_kernel, stride = conv_stride, padding = conv_pad),
            torch.nn.BatchNorm1d(channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels = channel, out_channels = channel, kernel_size = conv_kernel, stride = conv_stride, padding = conv_pad),
            torch.nn.BatchNorm1d(channel),
        )
        self.bn = torch.nn.BatchNorm1d(channel)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.res(x)
        if x.shape[1] == identity.shape[1]:
            x += identity
        # repeat the smaller block till it reaches the size of the bigger block
        elif x.shape[1] > identity.shape[1]:
            if x.shape[1] % identity.shape[1] == 0:
                x += identity.repeat(1, x.shape[1]//identity.shape[1], 1)
            else:
                raise RuntimeError("Dims in ResBlock needs to be divisible on the previous dims!!")
        else:
            if identity.shape[1] % x.shape[1] == 0:
                identity += x.repeat(1, identity.shape[1]//x.shape[1], 1)
            else:
                raise RuntimeError("Dims in ResBlock needs to be divisible on the previous dims!!")
            x = identity
        x = self.bn(x)
        x = self.relu(x)
        return x

class CNNRes(torch.nn.Module):

    def __init__(self, channels, conv_kernels, conv_strides, conv_padding, pool_padding, num_classes=5):
        assert len(conv_kernels) == len(channels) == len(conv_strides) == len(conv_padding)
        super(CNNRes, self).__init__()

        # create conv block
        prev_channel = 1
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=prev_channel, out_channels=channels[0][0], kernel_size=conv_kernels[0],
                            stride=conv_strides[0], padding=conv_padding[0]),
            # add batch norm layer
            torch.nn.BatchNorm1d(channels[0][0]),
            # adding ReLU
            torch.nn.ReLU(),
            # adding max pool
            torch.nn.MaxPool1d(kernel_size=4, stride=4, padding=pool_padding[0]),
        )

        # create res
        prev_channel = channels[0][0]
        self.res_blocks = torch.nn.ModuleList()
        for i in range(1, len(channels)):
            # add stacked res layer
            block = []
            for j, conv_channel in enumerate(channels[i]):
                block.append(ResBlock(prev_channel, conv_channel, conv_kernels[i], conv_strides[i], conv_padding[i]))
                prev_channel = conv_channel
            self.res_blocks.append(torch.nn.Sequential(*block))

        # create pool blocks
        self.pool_blocks = torch.nn.ModuleList()
        for i in range(1, len(pool_padding)):
            # adding Max Pool (drops dims by a factor of 4)
            self.pool_blocks.append(torch.nn.MaxPool1d(kernel_size=4, stride=4, padding=pool_padding[i]))

        # global pooling
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.linear = torch.nn.Linear(prev_channel, num_classes)

    def forward(self, inwav):
        inwav = self.conv_block(inwav)
        for i in range(len(self.res_blocks)):
            # apply conv layer
            inwav = self.res_blocks[i](inwav)
            # apply max_pool
            if i < len(self.pool_blocks): inwav = self.pool_blocks[i](inwav)
        # apply global pooling
        out = self.global_pool(inwav).squeeze()
        out = self.linear(out)
        return out.squeeze()

Cnn_res = CNNRes(channels = [[48], [48]*3, [96]*4, [192]*6, [384]*3],
          conv_kernels = [80, 3, 3, 3, 3],
          conv_strides = [4, 1, 1, 1, 1],
          conv_padding = [38, 1, 1, 1, 1],
          pool_padding = [0, 0, 0, 2])

class CnnLSTM(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,layer_dim,out_channels):
        super(CnnLSTM, self).__init__()
        self.lstm = nn.LSTM(embedding_dim,hidden_dim,layer_dim,batch_first=True,bidirectional=False)
        # 一维卷积神经网络
        self.conv1 = nn.Conv1d(1, out_channels=out_channels, kernel_size=2,stride=2)
        self.BN1 = torch.nn.BatchNorm1d(out_channels)
        self.max_pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, out_channels=32, kernel_size=2, stride=2)
        self.BN2 = torch.nn.BatchNorm1d(32)
        self.max_pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(32, out_channels=64, kernel_size=2, stride=2)
        self.BN3 = torch.nn.BatchNorm1d(64)
        self.max_pool3 = nn.MaxPool1d(2)
        # self.conv4 = nn.Conv1d(64, out_channels=128, kernel_size=8, stride=2)
        # self.BN4 = torch.nn.BatchNorm1d(128)
        # self.max_pool4 = nn.MaxPool1d(2)
        # self.conv5 = nn.Conv1d(128, out_channels=256, kernel_size=4, stride=2)
        # self.BN5 = torch.nn.BatchNorm1d(256)
        # self.max_pool5 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * 39,  120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.conv1(x)
        x = self.max_pool1(self.BN1(F.tanh(x)))
        x = self.conv2(x)
        x = self.max_pool2(self.BN2(F.tanh(x)))
        x = self.conv3(x)
        x = self.max_pool3(self.BN3(F.tanh(x)))
        # x = self.conv4(x)
        # x = self.max_pool4(self.BN4(F.tanh(x)))
        # x = self.conv5(x)
        # x = self.max_pool5(self.BN5(F.tanh(x)))
        x = x.view(-1, 39 * 64)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
