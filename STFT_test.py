#https://blog.csdn.net/qq_27825451/article/details/88553441
import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
# 原始信号的两个正弦波的频率分别为，100Hz、200Hz,最大频率为400赫兹。
# 根据采样定理，fs至少是200赫兹的2倍，这里选择500赫兹，即在一秒内选择500个点
fs = 100 # 采样频率fs=500Hz
N = 200 # 采样点数
n = np.arange(0,N)
t = np.arange(0, N) * (1.0 / fs)
# t = np.linspace(0,2,500)
y = 0.5*np.sin(2*np.pi*15*t)+2*np.sin(2*np.pi*40*t)
print(len(y))
plt.plot(t, y)
plt.title("时域信号")
plt.show()

y_fft = fft(y)
mag = np.abs(y_fft)
f = n * fs / N  #频率序列
plt.plot(f[1:int(N/2)+1],mag[1:int(N/2)+1]) # 绘出随频率变化的振幅
plt.xlabel('freq/Hz');
plt.ylabel('amplitude')
plt.title("幅度谱") # fs=100Hz，Nyquist频率为fs/2=50Hz。
plt.show()
# 相谱(相位谱和频率普是回事儿，想着把频谱中的幅值部分换成相角就可以了）
phase = np.angle(y_fft)      # 求得Fourier变换后的振幅
# 第 n个数的频率 fs / fft采样点数 * 第n个数
f = fs * n / N    # 频率序列
plt.plot(f[1:int(N/2)+1],phase[1:int(N/2)+1])
plt.xlabel('freq/hz');
plt.ylabel('phase')
plt.title("相位谱")
plt.show()
plt.tight_layout()
#  x(m)对应第m个采样点
# 计算语谱时采用不同窗长度，可以得到两种语谱图，即窄带和宽带语谱图。
# 长时窗（至少两个基音周期）常被用于计算窄带语谱图，短窗则用于计算宽带语谱图。
# 窄带语谱图具有较高的频率分辨率和较低的时间分辨率，良好的频率分辨率可以让语音的每个谐波分量更容易被辨别，在语谱图上显示为水平条纹。
# 宽带语谱图具有较高的时间分辨率和较低的频率分辨率，低频率分辨率只能得到谱包络，良好的时间分辨率适合用于分析和检验英语语音的发音
