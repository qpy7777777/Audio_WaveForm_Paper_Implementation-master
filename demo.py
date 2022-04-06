# 滤波器
import torch
import torch.nn as nn
import torch.nn.functional as F
# a = torch.Tensor([[2,1,6]])
# # softmax
# print(F.softmax(a,dim=1))
#
# # logsoftmax
# logsoftmax = nn.LogSoftmax()
# print(logsoftmax(a))
# # 负对数似然损失
# target1=torch.Tensor([0]).long()
# target2 = torch.Tensor([1]).long()
# target3 = torch.Tensor([2]).long()
# print(target1,target2,target3)
# # 测试   取出a中对应target位置的值并取负号
# n1 = F.nll_loss(a,target1)
# print(n1)
# n2 = F.nll_loss(a,target2)
# print(n2)
# n3 = F.nll_loss(a,target3)
# print(n3)
#
# # # nn.CrossEntropy()是nn.LogSoftmax()和nn.NLLLoss的结合
# n4 = F.cross_entropy(a,target3)
# print(n4)
#分类正确的是正样本，分类错误的是负样本。用来增加模型的健壮性

import torch
import torch.nn as nn
import torch.nn.functional as F
class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 5, size_average=True):

        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1) # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction
    def forward(self, logits, target):
        # logits: [N, C, H, W], target: [N, H, W]
        # loss = sum(-y_i * log(c_i))
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # [N, C, HW]
            logits = logits.transpose(1, 2)   # [N, HW, C]
            logits = logits.contiguous().view(-1, logits.size(2))    # [NHW, C]
        target = target.view(-1, 1)    # [NHW，1]

        logits = F.log_softmax(logits, 1)
        print(logits)
        logits = logits.gather(1, target)   # [NHW, 1]
        loss = -1 * logits

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
class CrossEntropyFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 5,reduction='mean'):
        super(CrossEntropyFocalLoss, self).__init__()
        self.reduction = reduction
        if isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, logits, target):
        # logits: [N, C, H, W], target: [N, H, W]
        # loss = sum(-y_i * log(c_i))
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # [N, C, HW]
            logits = logits.transpose(1, 2)  # [N, HW, C]
            logits = logits.contiguous().view(-1, logits.size(2))  # [NHW, C]
        target = target.view(-1, 1)  # [NHW，1]

        pt = F.softmax(logits, 1)
        pt = pt.gather(1, target).view(-1)  # [NHW]
        log_gt = torch.log(pt)

        if self.alpha is not None:
            # alpha: [C]
            alpha = self.alpha.gather(0, target.view(-1))  # [NHW]
            log_gt = log_gt * alpha

        loss = -1 * (1 - pt) ** self.gamma * log_gt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
# input is of size N x C = 3 x 5
input = torch.randn(3,5,requires_grad=True)
target = torch.tensor([1,0,4])
print(input)
print(target)
# softmax = F.softmax(input,dim=1)
# print("softmax",softmax)
# m = F.log_softmax(input,dim=1)
# print("logsoftmax",m)
# loss = nn.NLLLoss()
# output = loss(m,target)
# print(output)
# print(F.cross_entropy(input,target))
# print(F.nll_loss(input,target))
# print(F.cross_entropy(input, target))
# loss_function = CrossEntropyLoss()
# loss= loss_function(input,target)
# print(loss)
# loss_focal = CrossEntropyFocalLoss()
# focal_loss = loss_focal(input,target)
# print(focal_loss)
import math
# print(math.e ** (0) * 2)
import wave
import numpy as np
audio_list = ["data/0/data_CH1_1.wav","data/1/data_CH1.wav","data/2/5511300F.wav",
              "data/3/5702100A.wav","data/4/7202100Q.wav"]
file = wave.open(audio_list[0], 'r')
params = file.getparams()
# 声道，采样宽度，帧速率，帧数，唯一标识，无损
nchannels, sampwidth, framerate, nframes = params[:4]
# print(nchannels,sampwidth, framerate, nframes)
str_data = file.readframes(nframes)
file.close()
str_data = np.frombuffer(str_data, dtype=np.short)
temp_data = str_data.T
print(str_data[:10])
print(temp_data[:10])
    # 1 2 17067 4266751
    # 1 2 17067 5120100
    # 1 2 14900 354172
    # 1 2 30000 35987
    # 1 2 10240 29040
import os
import glob
file_path = "data"
cate = [file_path + "/" + x for x in os.listdir(file_path) if os.path.isdir(file_path + "/" + x)]
labels = []
audio_name = []
for idx, folder in enumerate(cate):
    # print(folder) data/0
    for audio in glob.glob(folder + '/*.wav'):
        audio_name.append(os.path.join(audio,"\n"))
        x = os.path.basename(folder)
        labels.append(os.path.join(x,"\n"))
with open("labels.txt","w") as f:
    f.write(''.join(labels))


