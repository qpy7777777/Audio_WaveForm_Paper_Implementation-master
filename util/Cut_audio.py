import os
import glob
import wave
import numpy as np

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        print("success")

# 裁剪数据
def Cut_file(path):
    folder = [path + "/" + x for x in os.listdir(path) if os.path.isdir(path+ "/" + x)]
    for index, file_dir in enumerate(folder):
        for audio in glob.glob(file_dir+"/*.wav"):
            print("Cut file name is {}".format(audio))
            file = wave.open(audio, 'r')
            params = file.getparams()
            nchannels, sampwidth, framerate, nframes = params[:4]
            # 裁剪的总帧数
            CutFrameNum = int(framerate * cutTimeDef)
            str_data = file.readframes(nframes)
            file.close()
            str_data = np.frombuffer(str_data,dtype = np.short)
            if nchannels >1:
                str_data.shape = -1,2
                str_data = str_data.T
                temp_data = str_data.T
            else:
                str_data = str_data.T
                temp_data = str_data.T
            # print(len(temp_data))
            # print("裁剪数量：",nframes / CutFrameNum)
            StepNum = CutFrameNum # # 5120
            # print(StepNum)
            StepTotalNum = 0
            num = 0
            while StepTotalNum<nframes:
                # print("Step=%d" % num)
                FileName = os.path.basename(os.path.splitext(audio)[0]) + '_'+str(num+1)+'.wav'
                out_path = os.path.join(r"../cutesc_data",os.path.basename(file_dir),FileName)
                # print("outpath:",out_path)
                temp_dataTemp = temp_data[StepNum * (num):StepNum * (num + 1)]
                num = num+1
                StepTotalNum = num * StepNum
                # print(StepTotalNum)
                # print(os.path.dirname(out_path))
                create_folder(os.path.dirname(out_path))
                # temp_dataTemp = temp_dataTemp.astype(np.short)  # 打开WAV文档
                f = wave.open(out_path, "wb")  #
                # 配置声道数、量化位数和取样频率
                f.setnchannels(nchannels)
                f.setsampwidth(sampwidth)
                f.setframerate(framerate)
                # 将wav_data转换为二进制数据写入文件
                f.writeframes(temp_dataTemp.tostring())
                f.close()

if __name__ == "__main__":
    path = "../esc-5"
    cutTimeDef = 0.5  # 裁剪时间 0.5s
    Cut_file(path)
    print("Finishing")
