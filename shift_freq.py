import argparse

import librosa
import soundfile as sf
import numpy as np
import rir_generator as rir
from tqdm import tqdm
from scipy import signal
from scipy.fftpack import fft, ifft, fftshift

import pyHowling
from config import shift_freq_conf


class Shift_freq(object):
    def __init__(self, conf, args):
        # path
        self.clean_wav = args.clean_wav
        self.howl_wav = args.howl_wav
        self.pha_shift_wav = args.pha_shift_wav
        self.frq_shift_wav = args.frq_shift_wav

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

        # notch filter configure
        self.current_frame = np.zeros(self.win_len)

        # magaphone configure
        self.gain = conf['gain']
        self.N = conf['N']
        self.feedforward = np.zeros(conf['delay']).T
        self.feedforward[-1] = 1
        self.feedbackward = rir.generate(c=340, fs=self.sample_rates, r=self.receiver, s=self.speaker, L=self.room_size,
                                         reverberation_time=self.t60, nsample=self.rir_length)
        # self.pa, _ = signal.freqz([0, 0.48, 0.5, 1], [1, 1, 0, 0], worN=self.N)
        self.pa = np.zeros(self.N)
        self.pa[self.N // 2] = 1

        # hilbert
        self.order = 30
        self.pi = np.pi
        self.freq_shift = 5
        self.phase_shift = self.pi / 3
        self.hwin = np.blackman(self.order)
        self.Ifilter = np.zeros(self.order//2)

        # init
        temp1 = np.arange(-self.order / 2, self.order / 2 + 1, 1)
        temp1 = (1 - (-1) ** temp1) / temp1
        temp1[int(self.order/2)] = 0

        self.Ifilter[-1] = 1
        self.Qfilter = temp1 * (1 / self.pi) * np.blackman(self.order+1)

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
            howl[i] = max(-1, howl[i])


            xs2[1:] = xs2[: - 1]
            xs2[0] = howl[i]
            temp = np.dot(xs2.T, self.feedbackward)

        sf.write(self.howl_wav, howl, self.sample_rates)

    def pha_shift(self):
        # load wav file
        x, _ = librosa.load(self.clean_wav, sr=self.sample_rates)

        # temp
        temp = 0
        xs1 = np.zeros(self.feedforward.shape[0])
        xs2 = np.zeros(self.feedbackward.shape)
        xs3 = np.zeros(self.N)

        xs4 = np.zeros(self.order+1)
        xs5 = np.zeros(len(self.Ifilter))

        howl = np.zeros(len(x))

        for i in range(len(x)):
            xs1[1:] = xs1[: - 1]
            xs1[0] = x[i] + temp

            xs4[1:] = xs4[: - 1]
            xs4[0] = xs1[0]
            Q = np.dot(xs4.T, self.Qfilter)

            xs5[1:] = xs5[: - 1]
            xs5[0] = xs1[0]
            I = np.dot(xs5.T, self.Ifilter)

            xs1[0] = I * np.cos(self.phase_shift) - Q * np.sin(self.phase_shift)

            howl[i] = self.gain * np.dot(xs1.T, self.feedforward)

            xs3[1:] = xs3[: - 1]
            xs3[0] = howl[i]
            howl[i] = np.dot(xs3.T, self.pa)

            howl[i] = min(1, howl[i])
            howl[i] = max(-1, howl[i])

            xs2[1:] = xs2[: - 1]
            xs2[0] = howl[i]
            temp = np.dot(xs2.T, self.feedbackward)

        sf.write(self.pha_shift_wav, howl, self.sample_rates)


    def frq_shift(self):
        # load wav file
        x, _ = librosa.load(self.clean_wav, sr=self.sample_rates)

        # temp
        h = np.exp(2*np.pi*5*1j*np.arange(1, 202)/4)
        temp = 0
        xs1 = np.zeros(self.feedforward.shape[0])
        xs2 = np.zeros(self.feedbackward.shape)
        xs3 = np.zeros(self.N)
        xs4 = np.zeros(len(h))

        howl = np.zeros(len(x))

        for i in range(len(x)):
            xs1[1:] = xs1[: - 1]
            xs1[0] = x[i] + temp

            xs4[1:] = xs4[: - 1]
            xs4[0] = xs1[0]
            xs1[0] = np.dot(xs4.T, h)
            xs1[0] = xs1[0] * np.exp(2*np.pi*1j*i/self.sample_rates*self.freq_shift)
            xs1[0] = np.real(xs1[0])

            howl[i] = self.gain * np.dot(xs1.T, self.feedforward)

            xs3[1:] = xs3[: - 1]
            xs3[0] = howl[i]
            howl[i] = np.dot(xs3.T, self.pa)

            howl[i] = min(1, howl[i])
            howl[i] = max(-1, howl[i])

            xs2[1:] = xs2[: - 1]
            xs2[0] = howl[i]
            temp = np.dot(xs2.T, self.feedbackward)

        sf.write(self.frq_shift_wav, howl, self.sample_rates)


def main():
    # parse the configurations
    parser = argparse.ArgumentParser(description='Additioal configurations for training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--clean_wav',
                        type=str,
                        default=r'./data/Shift_Freq/SI1186.wav')

    parser.add_argument('--howl_wav',
                        type=str,
                        default=r'./data/Shift_Freq/SI1186_howl.wav')

    parser.add_argument('--pha_shift_wav',
                        type=str,
                        default=r'./data/Shift_Freq/SI1186_howl_suppress_pha.wav')

    parser.add_argument('--frq_shift_wav',
                        type=str,
                        default=r'./data/Shift_Freq/SI1186_howl_suppress_frq.wav')

    args = parser.parse_args()

    # shift freqs
    shift_freq = Shift_freq(shift_freq_conf, args)
    shift_freq.gen_howl()
    shift_freq.pha_shift()
    shift_freq.frq_shift()


if __name__ == '__main__':
    main()