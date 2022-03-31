import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
from experiments import test_data,test_loader
# 模型测试
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiments')
    parser.add_argument('--model', default="M5", type=str, help='The model name')
    net = torch.load(model+".pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    ##对测试集进行预测，并可视化预测结果
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            audio, labels = data
            audio = audio.unsqueeze(1)
            audio, labels= audio.to(device),labels.to(device)
            net.eval()
            outputs = net(audio)
            _,predicted=torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()

    print("Accuracy of the network on the test audio:%d %%" % (100* correct/total))