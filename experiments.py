import argparse
import torch
import os
import matplotlib.pyplot as plt
from model.myModel import *
from torch.utils.data import Dataset
from util.preprocess import SoundDataset
from main_code import key_func

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiments')
    parser.add_argument('--model', default="M5", type=str, help='The model name')
    parser.add_argument('--batchSize', default=8, type=int, help='Batch size to use for train/test sets')

    args = parser.parse_args()
    if args.model == "M5":
        model = M5()
    elif args.model == "M18":
        model = M18()
    elif args.model == "CNNRes":
        model = Cnn_res
    elif args.model == "RCNN":
        # 初始化模型
        input_size = 5000
        hidden_dim = 5200
        layer_dim = 2
        out_channels = 16
        model = CnnLSTM(input_size, hidden_dim, layer_dim, out_channels)

    batch_size = args.batchSize

    # 数据路径
    root_data_dir = r"12_new/"

    # 读取文件
    train_data = SoundDataset((os.path.join(root_data_dir, "train")))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batchSize,
                                               num_workers=0, shuffle=True)
    print("train_loader的batch数量为：", len(train_loader))

    test_data = SoundDataset((os.path.join(root_data_dir, "test")))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batchSize, num_workers=0)

    # apply initializer
    model.apply(init_weights)
    # 打印网络参数数量
    print("Num Parameters:", sum([p.numel() for p in model.parameters()]))

    # create criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.0001)  # by default, l2 regularization is implemented in the weight decay.
    model_ft, train_process = key_func(model, 0.8, criterion, train_loader, test_loader,optimizer, EPOCH=5)
    torch.save(model_ft, args.model+'.pt')

    ##可视化模型训练过程
    plt.figure(figsize=(12, 4))
    ##损失函数
    plt.subplot(1, 2, 1)
    plt.plot(train_process.epoch, train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process.epoch, train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    ##精度
    plt.subplot(1, 2, 2)
    plt.plot(train_process.epoch, train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process.epoch, train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()

