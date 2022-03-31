import librosa.display
import torch
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import pandas as pd
import os
import time
import copy
from tensorboardX import SummaryWriter

# CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sanity check

# Focal Loss损失函数设计
# class focal_loss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2, num_classes = 5, size_average=True):
#         """
#         focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
#         步骤详细的实现了 focal_loss损失函数.
#         :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
#         :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
#         :param num_classes:     类别数量
#         :param size_average:    损失计算方式,默认取均值
#         """
#         super(focal_loss,self).__init__()
#         self.size_average = size_average
#         if isinstance(alpha,list):
#             assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
#             print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
#             self.alpha = torch.Tensor(alpha)
#         else:
#             assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
#             print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
#             self.alpha = torch.zeros(num_classes)
#             self.alpha[0] += alpha
#             self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
#         self.gamma = gamma
#
#     def forward(self, preds, labels):
#         """
#         focal_loss损失计算
#         :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
#         :param labels:  实际类别. size:[B,N] or [B]
#         :return:
#         """
#         # assert preds.dim()==2 and labels.dim()==1
#         preds = preds.view(-1,preds.size(-1))
#         self.alpha = self.alpha.to(preds.device)
#         preds_softmax = F.softmax(preds, dim=1) # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
#         preds_logsoft = torch.log(preds_softmax)
#         preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
#         preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
#         self.alpha = self.alpha.gather(0,labels.view(-1))
#         loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
#         loss = torch.mul(self.alpha, loss.t())
#         if self.size_average:
#             loss = loss.mean()
#         else:
#             loss = loss.sum()
#         return loss
def key_func(model, train_rate,criterion, train_loader,test_loader,optimizer, EPOCH):
    # 获得一个batch的数据
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    for step, (b_x, b_y) in enumerate(train_loader):  # torch.Size([64, 1, 28, 28])
        if step > 0:
            break
        # 可视化一个batch的图像
        batch_x = b_x.squeeze(1).numpy()
        batch_y = b_y.numpy()
        print(batch_y)
        plt.figure(figsize=(12, 5))
        for ii in range(len(batch_y)):
            plt.subplot(4, 2, ii + 1)
            librosa.display.waveplot(batch_x[ii, :], sr=10000)
            plt.title(batch_y[ii], size=9)
            plt.axis("off")
            plt.subplots_adjust(wspace=0.05)
        plt.savefig("plot.pdf")
        plt.show()
    # 打印日志
    log_file = './log.txt'
    writer = SummaryWriter()

    since = time.time()
    ##计算训练使用的batch数量
    batch_num = len(train_loader)
    train_batch_num = round(batch_num * train_rate)
    # 复制模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    for epoch in range(EPOCH):
        print('Epoch {}/{}'.format(epoch, EPOCH - 1))
        print('-' * 10)
        ##每个epoch有两个训练阶段
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        # Iterate over data.
        for i,(audio,label) in enumerate(train_loader):
            if i < train_batch_num:
                scheduler.step()
                # 如果模型中有BN层(Batch Normalization）和 Dropout，需要在训练时添加model.train()。
                model.train()  # Set model to training mode
                audio = audio.unsqueeze(1)
                # print(audio.shape) #[8,1,5000]
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                outputs = model(audio)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, label)
                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()
                # statistics
                train_loss += loss.item() * audio.size(0)
                train_corrects += torch.sum(preds == label)
                train_num += audio.size(0)

            else:
                model.eval()  # 3设置模型为评估模式      # 如果模型中有BN层(Batch Normalization）和Dropout，在测试时添加model.eval()
                audio = audio.unsqueeze(1)
                output = model(audio)
                pre_lab = torch.argmax(output, 1)
                loss = criterion(output, label)
                val_loss += loss.item() * audio.size(0)
                val_corrects += torch.sum(pre_lab == label)
                val_num += audio.size(0)
        print(train_loss, train_num, train_corrects)
        print(val_loss,val_num,val_corrects)
        ##计算一个epoch在训练集和验证集上的损失和精度
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        print('{} Train Loss:{:.4f}  Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss:{:.4f}  Val Acc:{:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))
        ##拷贝模型最高精度下的参数
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        time_use = time.time() - since
        print("Train and val complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))
    ##使用最好模型的参数
    model.load_state_dict(best_model_wts)
    train_process = pd.DataFrame(
        data={
            "epoch": range(EPOCH),
            "train_loss_all": train_loss_all,
            "val_loss_all": val_loss_all,
            "train_acc_all": train_acc_all,
            "val_acc_all": val_acc_all
        }
    )
    return model, train_process




