# 数据划分
# 将0和1的音频分别移动到训练集和验证集中，其中70%的数据作为训练集，10%的音频作为验证集，使用shutil.move（）来移动音频。
# 新建文件夹train,test,将数据集放入train中，利用代码将30%的数据移动到test中
import os
import shutil
import glob
import numpy as np

# 测试比例
test_ratio = 0.7

def split_file(old_path, new_path, choice):
    # 读取目录文件
    file = [old_path + '/' + x for x in os.listdir(old_path) if os.path.isdir(old_path + '/' + x)]

    # 划分文件数据集
    for i in range(len(file)):  # len(file) = 2
        list_dir = os.listdir(file[i])  # [0.jpg,1,jpg,,...9.jpg]
        if choice == 'train':
            num = int(test_ratio * len(os.listdir(file[i])))  # num = 9
        else:
            # 剩余的作为测试集
            num = len(os.listdir(file[i]))

        # 随机打乱数据
        index = np.random.choice(np.arange(num), size=num, replace=False)
        #         print(index)

        # 移动数据
        for j in range(num):
            old_name = os.path.join(file[i], list_dir[index[j]])
            #             print(old_name)
            new_file = os.path.join(new_path, os.path.basename(file[i]))
            if not os.path.exists(new_file):
                os.makedirs(new_file)
            shutil.move(old_name, new_file)


path = r"../cutesc_data"

for choice in ["train", "test"]:
    # 新建换分数据之后的文件夹
    if choice == "train":
        new_path = r"../12_new/train"
    else:
        new_path = r"../12_new/test"
    # 划分数据集
    split_file(path, new_path, choice)