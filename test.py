import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 这里导入你自己的数据
# x_axix，train_pn_dis这些都是长度相同的List()
x_axix = list(range(60))
x_axix = np.array(x_axix)
train_acys = []
test_acys = []
pn_dis = []
thresholds = []
for i in range(len(x_axix)):
    num = np.random.uniform(0.5, 1.0)
    train_acys.append(num)
train_acys = np.array(train_acys)
print(train_acys.shape)
for i in range(len(x_axix)):
    num = np.random.uniform(0.6,0.9)
    test_acys.append(num)
test_acys = np.array(test_acys)
for i in range(len(x_axix)):
    num = np.random.uniform(0.0,0.1)
    pn_dis.append(num)
pn_dis = np.array(pn_dis)
for i in range(len(x_axix)):
    num = np.random.uniform(0.1,0.2)
    thresholds.append(num)
thresholds = np.array(thresholds)

#开始画图
# sub_axix=filter(lambda x: x%2==0,x_axix)
plt.title("Result Analysis")
plt.plot(x_axix,train_acys,color='green',label='training accuracy')
plt.plot(x_axix,test_acys,color='red',label='testing accuracy')
plt.plot(x_axix,pn_dis,color='skyblue',label='PN distance')
plt.plot(x_axix,thresholds,color='blue',label="threshold")
plt.legend() #显示图例
plt.xlabel('iteration times')
plt.ylabel('rate')
plt.show()
#python 一个折装图绘制多个感转