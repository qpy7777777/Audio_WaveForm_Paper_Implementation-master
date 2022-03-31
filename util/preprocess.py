import librosa
from warnings import filterwarnings
filterwarnings('ignore')
import os, glob
from torch.utils.data import Dataset
import numpy as np
import torch

nw = 441 # 帧长10ms
inc = 110 # 帧移四分之一帧长
# 信号帧数fn：fn =（N-wlen）/inc+1 397
# 分帧
def enframe(signal, nw, inc):
    '''将音频信号转化为帧。
    参数含义：
    signal:原始音频型号
    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    inc:相邻帧的间隔（同上定义）
    '''
    signal_length = len(signal)  # 信号总长度
    #     print("信号原始长度",signal_length) #5120100
    if signal_length < nw:
        nf = 1
    else:
        nf = int(np.ceil((1.0 * signal_length - nw + inc) / inc))
    pad_length = int((nf - 1) * inc + nw)  # #所有帧加起来总的铺平后的长度
    zeros = np.zeros((pad_length - signal_length,))  # #不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal = np.concatenate((signal, zeros))  # #填补后的信号记为pad_signal np.ceil向上取整，会导致实际分帧后的长度大于信号本身的长度
    # print("pad_signal", pad_signal.shape)  # (5120128,)
    indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc),
                                                           (nw, 1)).T  # 相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
    indices = np.array(indices, dtype=np.int32)  # 将indices转换为矩阵
    frames = pad_signal[indices]  # 得到帧信号 #(39998,512)
    #     print("frames",frames[:2])
    #    win=np.tile(winfunc(nw),(nf,1))  #window窗函数，这里默认取1
    #    return frames*win   #返回帧信号矩阵  加窗的语音分帧
    return frames

# 对数据进行归一化处理
def normalization(data):
    max_data = np.max(data)
    min_data = np.min(data)
    new_audio = (data-min_data)/(max_data-min_data)
    return new_audio

# normalize audio signal to have mean=0 & std=1
def Normalize(waveform):
        return (waveform-waveform.mean()) / waveform.std()

# 构建数据集
class SoundDataset(Dataset):
    def __init__(self, file_path):
        cate = [file_path + "/" + x for x in os.listdir(file_path) if os.path.isdir(file_path + "/" + x)]
        self.labels = []
        self.audio_name = []
        for idx, folder in enumerate(cate):
            for audio in glob.glob(folder + '/*.wav'):
                self.audio_name.append(audio)
                x = os.path.basename(folder)
                self.labels.append((int)(x))
        self.file_path = file_path

    def __getitem__(self, index):
        audio = self.audio_name[index]  # + '.wav'
        sound, sample = librosa.load(audio,sr=10000)#重采样
        sound = Normalize(sound)
        soundData = torch.Tensor(sound)
        return soundData, self.labels[index]

    def __len__(self):
        return len(self.audio_name)


