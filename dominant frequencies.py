import os
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
from scipy.signal import periodogram
# 查看据集类别主频率的分布图
#Using fft to find the dominant frequencies(this operation can be parallelised)
# [classID] = a numeric identifier of the sound class (see description of classID below for further details)
#         A numeric identifier of the sound class:ESC-50
#         0 = airplane
#         1 = breathing
#         2 = brushing_teeth
#         3 = can_opening
#         4 = car_horn]
freqs = {k:[] for k in range(5)}
for folder in os.listdir('12_new/train'):
    for file in os.listdir('12_new/train'+'/'+folder):
        class_id = os.path.basename(folder)
        data,sample_rate = librosa.load('12_new/train'+'/'+folder + '/'+file)
        freq,PSD = periodogram(data,fs=sample_rate)
        max_id = np.flip(np.argsort(PSD))[:1][0]
        freqs[int(class_id)].append(freq[max_id])
    print('------Done for a folder..!----')
#%%
import seaborn as sns
#distribution plots
#for class 0
sns.distplot(freqs[0],kde=True,label='Dominant frequency distribution of class 0')
plt.show()
#for class 1
sns.distplot(freqs[1],kde=True,label='Dominant frequency distribution of class 1')
plt.show()
#for class 2
sns.distplot(freqs[2],kde=True,label='Dominant frequency distribution of class 2')
plt.show()
#for class 3
sns.distplot(freqs[3],kde=True,label='Dominant frequency distribution of class 3')
plt.show()
#for class 4
sns.distplot(freqs[4],kde=True,label='Dominant frequency distribution of class 4')
plt.show()
