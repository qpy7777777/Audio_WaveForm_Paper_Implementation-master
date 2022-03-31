import librosa.display
import torch
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import pandas as pd
import os
import time
import copy
import numpy as np
from tensorboardX import SummaryWriter
import seaborn as sns
# CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sanity check

def key_func(model, train_rate,criterion, train_loader,test_loader,optimizer, EPOCH):
    # 获得一个batch的数据
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    for step, (b_x, b_y) in enumerate(train_loader):  # torch.Size([64, 1, 28, 28])
        if step > 0:
            break
        # 可视化一个batch的图像
        batch_x = b_x.squeeze(1).numpy()
        batch_y = b_y.numpy()
        # print(batch_x.shape) #(8, 5000)
        plt.figure(figsize=(12, 5))
        for ii in range(len(batch_y)):
            plt.subplot(4, 2, ii + 1)
            # print(batch_x[ii, :].shape) #(5000,)
            time_wave = np.arange(0, batch_x.shape[1]) / 10000
            plt.plot(time_wave, batch_x[ii,:])
            plt.title(batch_y[ii], size=9)
            plt.axis("off")
            plt.subplots_adjust(wspace=0.05)
        plt.savefig("plot.pdf")
        plt.show()
        # 可视化一个batch,将每列特征变量使用箱线图进行显示，对比不同类别的邮件在每个特制变量上的数据分布情况
        # colname = spam.columns.values[:-1]
        # print(len(colname), type(colname))
        plt.figure(figsize=(20, 14))
        for ii in range(36):
            plt.subplot(6, 6, ii + 1)
            sns.boxplot(x=batch_y, y=batch_x[:, ii])
            plt.title(ii)
        plt.subplots_adjust(hspace=0.6)
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




