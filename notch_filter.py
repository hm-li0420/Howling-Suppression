import argparse

import librosa
import soundfile as sf
import numpy as np
import rir_generator as rir
from tqdm import tqdm
from scipy import signal
from scipy.fftpack import fft, ifft, fftshift

import pyHowling
from config import notch_filter_conf


class Notch_filter(object):
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

    def howling_detect(self, frame, win, nFFT, Slen, candidates, frame_id):
        insign = win * frame
        spec = np.fft.fft(insign, nFFT, axis=0)

        # ==========  Howling Detection Stage =====================#
        ptpr_idx = pyHowling.ptpr(spec[:Slen], 5)
        papr_idx, papr = pyHowling.papr(spec[:Slen], 5)
        pnpr_idx = pyHowling.pnpr(spec[:Slen], 10)
        intersec_idx = np.intersect1d(ptpr_idx, np.intersect1d(papr_idx, pnpr_idx))
        # print("papr:",papr_idx)
        # print("pnpr:",pnpr_idx)
        # print("intersection:", intersec_idx)
        for idx in intersec_idx:
            candidates[idx][frame_id] = 1
        ipmp = pyHowling.ipmp(candidates, frame_id)
        # print("ipmp:",ipmp)
        result = pyHowling.screening(spec, ipmp)
        # print("result:", result)
        return result

    def suppress_howl(self):
        # load wav file
        x, _ = librosa.load(self.clean_wav, sr=self.sample_rates)

        # temp
        b = [1.0, 0, 0]
        a = [0, 0, 0]
        pos = 0
        temp = 0
        frame_id = 0
        notch_freqs = []
        freqs = np.linspace(0, self.sample_rates // 2, self.win_len)
        Nframes = (len(x) - self.win_len + self.win_inc) // self.win_inc
        candidates = np.zeros([self.win_len, Nframes + 1], dtype='int')

        # buf
        xs1 = np.zeros(self.feedforward.shape[0])
        xs2 = np.zeros(self.feedbackward.shape[0])
        xs3 = np.zeros(self.N)
        howl = np.zeros(len(x))

        # suppress

        for i in range(len(x)):
            xs1[1:] = xs1[: - 1]
            xs1[0] = x[i] + temp
            self.current_frame[pos] = xs1[0]
            pos += 1
            if pos == self.win_len:
                # update notch filter frame by frame
                freq_ids = self.howling_detect(self.current_frame, self.win, self.win_len, self.win_len, candidates, frame_id)
                if (len(freq_ids) > 0 and (len(freq_ids) != len(notch_freqs) or not np.all(np.equal(notch_freqs, freqs[freq_ids])))):
                    notch_freqs = freqs[freq_ids]
                    sos = np.zeros([len(notch_freqs), 6])
                    for i in range(len(notch_freqs)):
                        b0, a0 = signal.iirnotch(notch_freqs[i], 1, self.sample_rates)
                        sos[i, :] = np.append(b0, a0)
                    b, a = signal.sos2tf(sos)
                self.current_frame[:self.win_len - self.win_inc] = self.current_frame[self.win_inc:]
                pos = self.win_inc
                frame_id = frame_id + 1

            howl[i] = self.gain * np.dot(xs1.T, self.feedforward)

            xs3[1:] = xs3[: - 1]
            xs3[0] = howl[i]
            howl[i] = np.dot(xs3.T, self.pa)
            howl[i] = min(1, howl[i])
            howl[i] = max(-1, howl[i])
            howl[i] = np.dot(xs3[:len(b)], b) - np.dot(xs2[:len(a) - 1], a[1:])

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
                        default=r'D:\桌面\数字助听器\python_howling_suppression-master\data\Notch_Filter\SI1186.wav')

    parser.add_argument('--howl_wav',
                        type=str,
                        default=r'D:\桌面\数字助听器\python_howling_suppression-master\data\Notch_Filter\SI1186_howl.wav')

    parser.add_argument('--suppress_wav',
                        type=str,
                        default=r'D:\桌面\数字助听器\python_howling_suppression-master\data\Notch_Filter\SI1186_howl_suppress.wav')


    args = parser.parse_args()

    notch_filter = Notch_filter(notch_filter_conf, args)
    notch_filter.gen_howl()
    notch_filter.suppress_howl()


if __name__ == '__main__':
    main()

