# 获取分类层可视化
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from util.preprocess import SoundDataset

audios = []
classes = []
# 数据路径
root_data_dir = r"12_new/"

# 读取文件
train_data = SoundDataset((os.path.join(root_data_dir, "train")))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batchSize,
                                           num_workers=0, shuffle=True)
print("train_loader的batch数量为：", len(train_loader))

test_data = SoundDataset((os.path.join(root_data_dir, "test")))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batchSize, num_workers=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiments')
    parser.add_argument('--model', default="M5", type=str, help='The model name')
    net = torch.load(model+".pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    for audio, label in train_loader:
        audio = audio.unsqueeze(1)
        out = net(audio)
        out = out.squeeze_(0)
        out = out.cpu().data.numpy()
        audios.append(out)
        classes.append(label)
    audios = np.asarray(audios)
    audios = TSNE(n_components=2, perplexity=30).fit_transform(audios)
    plt.figure(figsize=(6, 5))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'grey', 'orange', 'purple']
    for i in range(audios.shape[0]):
        plt.scatter(audios[i, 0], audios[i, 1], c=colors[classes[i]], s=20, marker='.')
    plt.savefig('tsne.png')



