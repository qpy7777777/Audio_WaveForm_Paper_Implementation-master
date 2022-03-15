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
# 多条线一起画
# plot([x],y,[fmt],[x2],y2,[fmt2],...,**kwargs)
# 可选参数[fmt]是一个字符串来定义图的基本属性如：颜色，点型，线型
# fmt接收的是每个属性的单个字母缩写，例如：plot(x,y,"bo-")
# 若属性用的是全名则不能用*fmt*参数来组合赋值，应该用关键字参数对单个属性赋值如：
#
# plot(x,y2,color='green', marker='o', linestyle='dashed', linewidth=1, markersize=6)
#
# plot(x,y3,color='#900302',marker='+',linestyle='-')

# 原文链接：https://blog.csdn.net/sinat_36219858/article/details/79800460
#开始画图
# sub_axix=filter(lambda x: x%2==0,x_axix)
plt.title("Result Analysis")
plt.plot(x_axix,train_acys,color='green',label='training acc')
plt.plot(x_axix,test_acys,color='red',label='testing acc')
plt.plot(x_axix,pn_dis,color='skyblue',label='PN distance')
plt.plot(x_axix,thresholds,color='blue',label="threshold")
plt.legend(loc='upper right') #显示图例
plt.xlabel('iteration times')
plt.ylabel('rate')
plt.show()
print("finish")
#python 一个折装图绘制多个感转
x=np.arange(0,2*np.pi,0.02)
y=np.sin(x)
y1=np.sin(2*x)
y2=np.sin(3*x)
ym1=np.ma.masked_where(y1>0.5,y1)
ym2=np.ma.masked_where(y2<-0.5,y2)
lines=plt.plot(x,y,x,ym1,x,ym2,'o')
#凝登统的离性
plt.setp(lines[0],linewidth=1)
plt.setp(lines[1],linewidth=2)
plt.setp(lines[2],linestyle='-',marker='',markersize=4)
#统的标备
plt.legend(("No mask","Masked if > 0.5","Masked if < -0.5"),loc='upper right')
plt.title("Masked line demo")
plt.show()