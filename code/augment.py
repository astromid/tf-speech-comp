import numpy as np
import librosa

L = 16000


class AudioAugmentor:

    def __init__(self, params, sample):
        self.sample = sample
        self.noise_samples = params['noise_samples']
        self.time_shift = params['time_shift']
        self.speed_tune = params['speed_tune']
        self.volume_tune = params['volume_tune']
        self.noise_vol = params['noise_vol']
        self.flags = np.random.rand(3)

    @staticmethod
    def _pad_sample(sample):
        n = len(sample)
        if n == L:
            return sample
        elif n < L:
            return np.pad(sample, (L - n, 0), 'constant', constant_values=0)
        else:
            begin = np.random.randint(0, n - L)
            return sample[begin:begin + L]

    @property
    def _get_silence(self):
        n = len(self.noise_samples)
        sample = self.noise_samples[np.random.randint(0, n)]
        return self._pad_sample(sample)

    @property
    def _time_shift(self):
        shift_ = int(np.random.uniform(-self.time_shift, self.time_shift))
        return np.roll(self.sample, shift_)

    @property
    def _speed_tune(self):
        rate_ = np.random.uniform(1 - self.speed_tune, 1 + self.speed_tune)
        return librosa.effects.time_stretch(self.sample.astype('float32'), rate_)

    @property
    def _get_noised(self):
        noise_ = self._get_silence
        volume_ = np.random.uniform(1 - self.volume_tune, 1 + self.volume_tune)
        noise_volume_ = np.random.uniform(0, self.noise_vol)
        return volume_ * self.sample + noise_volume_ * noise_

    @property
    def augmented(self):
        if self.flags[0] < 0.5:
            self.sample = self._time_shift
        if self.flags[1] < 0.5:
            self.sample = self._speed_tune
        self.sample = self._pad_sample(self.sample)
        if self.flags[2] < 0.5:
            self.sample = self._get_noised
        return self.sample

