import argparse

import librosa
import soundfile as sf
import numpy as np
import rir_generator as rir
from tqdm import tqdm
from scipy import signal
from scipy.fftpack import fft, ifft, fftshift

import pyHowling
from config import conf, adptive_filter_conf


class Adaptive_filter(object):
    def __init__(self, conf, args):
        # path
        self.clean_wav = args.clean_wav
        self.howl_wav = args.howl_wav
        self.suppress_wav = args.suppress_wav

        # audio configure
        self.sample_rates = conf['sample_rates']
        self.win_len = conf['win_len']
        self.win_inc = conf['win_inc']
        self.win_type = conf['win_type']
        self.win = np.hanning(self.win_len)

        # rir configure
        self.room_size = conf['room_size']
        self.receiver = conf['receiver']
        self.speaker = conf['speaker']
        self.t60 = conf['t60']
        self.rir_length = conf['rir_length']

        # af configure
        self.w = np.zeros(conf['M'])
        self.step = conf['step']
        self.eps = 1e-5

        # magephone configure
        self.gain = conf['gain']
        self.N = conf['N']
        self.feedforward = np.zeros(conf['delay']).T
        self.feedforward[-1] = 1
        self.feedbackward = rir.generate(c=340, fs=self.sample_rates, r=self.receiver, s=self.speaker, L=self.room_size,
                           reverberation_time=self.t60, nsample=self.rir_length)
        # self.pa, _ = signal.freqz([0, 0.48, 0.5, 1], [1, 1, 0, 0], worN=self.N)
        self.pa = np.zeros(self.N)
        self.pa[self.N//2] = 1

    def gen_howl(self):

        # load wav file
        x, _ = librosa.load(self.clean_wav, sr=self.sample_rates)

        # temp
        temp = 0
        xs1 = np.zeros(self.feedforward.shape[0])
        xs2 = np.zeros(self.feedbackward.shape)
        xs3 = np.zeros(self.N)
        howl = np.zeros(len(x))

        for i in range(len(x)):
            xs1[1:] = xs1[: - 1]
            xs1[0] = x[i] + temp

            howl[i] = self.gain * np.dot(xs1.T, self.feedforward)

            xs3[1:] = xs3[: - 1]
            xs3[0] = howl[i]
            howl[i] = np.dot(xs3.T, self.pa)

            howl[i] = min(1, howl[i])
            howl[i] = max(-1,howl[i])

            xs2[1:] = xs2[: - 1]
            xs2[0] = howl[i]
            temp = np.dot(xs2.T, self.feedbackward)

        sf.write(self.howl_wav, howl, self.sample_rates)

    def suppress_howl(self):
        # load wav file
        x, _ = librosa.load(self.clean_wav, sr=self.sample_rates)

        # temp
        temp = 0
        xs1 = np.zeros(self.feedforward.shape[0])
        xs2 = np.zeros(self.feedbackward.shape)
        xs3 = np.zeros(self.N)
        xs4 = np.zeros(len(self.w))
        howl = np.zeros(len(x))
        t = 0
        for i in range(len(x)):
            xs1[1:] = xs1[: - 1]
            xs1[0] = x[i] + temp

            xs4[1:] = xs4[: - 1]
            xs4[0] = xs1[0]
            xs1[0] = xs1[0] - np.dot(xs4.T, self.w)
            self.w = self.w + (self.step * xs1[0] * xs4) / (np.dot(xs4.T, xs4) + self.eps)

            howl[i] = self.gain * np.dot(xs1.T, self.feedforward)

            xs3[1:] = xs3[: - 1]
            xs3[0] = howl[i]
            howl[i] = np.dot(xs3.T, self.pa)

            howl[i] = min(1, howl[i])
            howl[i] = max(-1, howl[i])

            xs2[1:] = xs2[: - 1]
            xs2[0] = howl[i]
            temp = np.dot(xs2.T, self.feedbackward)

        sf.write(self.suppress_wav, howl, self.sample_rates)


def main():
    # parse the configurations
    parser = argparse.ArgumentParser(description='Additioal configurations for training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--clean_wav',
                        type=str,
                        default=r'D:\桌面\数字助听器\python_howling_suppression-master\data\Adaptive_Filter\SI1186.wav')

    parser.add_argument('--howl_wav',
                        type=str,
                        default=r'D:\桌面\数字助听器\python_howling_suppression-master\data\Adaptive_Filter\SI1186_howl.wav')

    parser.add_argument('--suppress_wav',
                        type=str,
                        default=r'D:\桌面\数字助听器\python_howling_suppression-master\data\Adaptive_Filter\SI1186_howl_suppress.wav')


    args = parser.parse_args()

    af = Adaptive_filter(adptive_filter_conf, args)
    af.gen_howl()
    af.suppress_howl()


if __name__ == '__main__':
    main()

