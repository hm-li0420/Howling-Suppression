# Howling-Suppression
用Python仿真啸叫的产生以及抑制，包括自适应法和陷波法


## 目录
1. [背景](#背景)
2. [安装](#安装)
3. [使用方法](#使用方法)
4. [参考](#参考)

## 背景
这是一个Python实现的啸叫抑制项目，整体项目分为三个部分：
1. 基于自适应法的啸叫抑制（adaptive_filter.py）
2. 基于啸叫检测和陷波器法的啸叫抑制（notch_filter.py）
3. 基于移频法的啸叫抑制（shift_freq.py）

## 使用方法
### 文件路径
```bash
HOWLING-SUPPRESSION
├─data
│  ├─Adaptive_Filter
│  ├─Notch_Filter
│  └─Shift_Freq
├─pyHowling
│    ├─__pycache__
│    ├─__init__.py
│    └─howling_detection.py
├─adaptive_filter.py
├─config.py
├─notch_filter.py
└─shift_freq.py
```
### 超参数设置
在config.py中可设置自适应法的超参数：adptive_filter_conf，陷波法的超参数：notch_filter_conf，移频法的超参数：shift_freq_conf  
以自适应法为例：
```bash
adptive_filter_conf = {
    # audio configure：帧长，帧移，fft长度，窗函数类型
    'sample_rates': 16000,
    'win_len': 320,
    'win_inc': 160,
    'fft_len': 480,
    'win_type': 'hann',

    # rir configure：房间大小，麦克风位置，音源位置，T60长度，rir长度
    'room_size': [10, 10, 10],
    'receiver': [3, 5, 1],
    'speaker': [3, 5.05, 1],
    't60': 0.3,
    'rir_length': 512,

    # megaphone configure：仿真助听器或扩声设备的延迟，增益
    'delay': 4,
    'gain': 0.6,
    'N': 201,

    # af configure：算法处理的长度M，自适应滤波器步进值，泄露系数设置
    'M': 64,
    'step': 0.002,
    'leak': 0
}
```
### 运行方法
1. 路径参数  
--clean_wav:干净语音用于仿真啸叫以及啸叫抑制之后的的音频   
--howl_wav:仿真啸叫产生后不经过抑制的音频  
--suppress_wav:仿真啸叫抑制之后产生的音频  
2. 运行基于自适应法的啸叫抑制  
```bash
python adptive_filter.py --clean_wav ./data/Adaptive_Filter/SI1186.wav --howl_wav ./data/Adaptive_Filter/SI1186_howl.wav --suppress_wav ./data/Adaptive_Filter/SI1186_howl_suppress.wav
```
3. 运行基于啸叫检测和陷波器法的啸叫抑制
```bash
python notch_filter.py --clean_wav ./data/Adaptive_Filter/SI1186.wav --howl_wav ./data/Adaptive_Filter/SI1186_howl.wav --suppress_wav ./data/Adaptive_Filter/SI1186_howl_suppress.wav
```
4. 运行移频法的啸叫抑制  
--pha_shift_wav:移相法结果  
--frq_shift_wav：移频法结果  
```bash
python notch_filter.py --clean_wav ./data/Adaptive_Filter/SI1186.wav --howl_wav ./data/Adaptive_Filter/SI1186_howl.wav --pha_shift_wav ./data/Shift_Freq/SI1186_howl_suppress_pha.wav --frq_shift_wav ./data/Shift_Freq/SI1186_howl_suppress_frq.wav
```
 
## 安装
Python环境大于3.6均可
```bash
$ git clone git@github.com:hm-li0420/Howling-Suppression.git
```

## 参考
[1] https://www.jianshu.com/p/2bb75b6f4c81  
[2] https://www.cnblogs.com/xingshansi/p/6862683.html  
[3] https://blog.csdn.net/zhoufan900428/article/details/9069475
[4] https://www.jianshu.com/p/779101548a83