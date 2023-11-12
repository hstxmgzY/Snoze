import librosa #音频处理
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.fftpack import dct
from matplotlib import cm
import os
def mfcc(data,sr):
    #预加重
    l = np.append(data[0],data[1:]-0.97*data[:-1])
    #分帧
    frame_size = 0.025 #30ms 为一帧
    frame_stride = 0.010 # 10ms 滑动（15ms的交叠部分）
    frame_length  = int(round(frame_size * sr)) #帧的大小
    frame_step = int(round(frame_stride * sr))  #帧的滑动
    num_frames = int(np.ceil(float(np.abs(len(data) - frame_length)) / frame_step)) #有多少帧
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
    np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    #indices 中的每一个元素为一帧,一帧有550个值，滑动220
    frames = l[np.mat(indices).astype(np.int32, copy=False)]
    #为frames添加窗函数
    frames *= np.hamming(frame_length)
    #傅里叶变换
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    #输出热力谱
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
    #转换mel频谱
    h = (2595 * np.log10(1 + (sr / 2) / 700)) #把最高频率转化为mel频率
    M = 40
    mel_points = np.linspace(0, h, M+2)  # 以mel频率为准划分坐标
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # 再把坐标的点转换为正常频率
    bin = np.floor((NFFT + 1) * hz_points / sr) #hz_point的索引
    fbank = np.zeros((M, int(np.floor(NFFT / 2 + 1)))) #一个40*257的容器
    #加三角窗
    for m in range(1, M + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    #对热力谱施加mel变换和三角窗
    filter_banks = np.dot(pow_frames, fbank.T)
    #数值稳定性
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    #以分贝为单位
    filter_banks = 20 * np.log10(filter_banks)
    #DCT变换
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : 13]
    #减去均值
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    # 画热力图
    plt.imshow(np.flipud(mfcc.T), cmap=plt.cm.jet, aspect=0.2, extent=[0, mfcc.shape[1], 0, mfcc.shape[0]])
    plt.show()
    return mfcc
#os.chdir("C:/Users/28314/Desktop/codescat/Python")
audio, sr = librosa.load('train.wav')
plt.figure(figsize=(8, 6))
plt.plot(audio)
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title("声音波形图")
plt.show()

mfccs = mfcc(audio, sr)
X = mfccs 
print('mfcc:', X)
plt.figure(figsize=(8, 6))
plt.plot(X)
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title("MFCC特征向量图")
plt.show()
kmeans = KMeans(n_clusters = 2) #创建k-means 聚类
kmeans.fit(X) 
labels = kmeans.labels_
train_data = X[labels == (1 - labels[0]) ] #将鼾声的信息作为训练集
#这里因为不知道鼾声段属于0还是1
#所以我们的处理是将鼾声段作为第一个数据取反的特征值
#因为第一段大概率不是鼾声，大概率是空白的噪音
model = GaussianMixture(n_components=2) #用高斯混合模型训练
model.fit(train_data)
probs = model.predict_proba(train_data) #计算训练集中每个样本生成的概率
probs = probs[np.argsort(probs[:,(1 - labels[0])]),:]    #升序排序
threshold = probs[int(len(probs) * 0.64)]

plt.figure(figsize=(8, 6))
plt.plot(probs)
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title("概率分布图")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title('Kmeans scatter')
plt.show()

data, fs = librosa.load('train.wav')
mfccs = mfcc(data, fs)
prob = model.predict_proba(mfccs)
cnt = 0
print('prob:',prob)
print('threshold:',threshold)
Anscnt = []
for i in range(len(prob)):
	if prob[i][1 - labels[0]] > threshold[1 - labels[0]]:
		Anscnt.append(1)
	else:
		Anscnt.append(0)  
for i in range(len(prob)):
    if Anscnt[i] == labels[i]:
        cnt += 1
print('正确率:',cnt / len(prob))
plt.figure(figsize=(8, 6))
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title("鼾声分布图")
plt.plot(Anscnt)
plt.show()