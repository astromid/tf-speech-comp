import numpy as np
import librosa

L = 16000


def augment(params, sample):

    noise_samples = params['noise_samples']
    time_shift = params['time_shift']
    speed_tune = params['speed_tune']
    volume_tune = params['volume_tune']
    noise_vol = params['noise_vol']
    flags = np.random.rand(3)

    if flags[0] < 0.5:
        shift_ = int(np.random.uniform(-time_shift, time_shift))
        sample = np.roll(sample, shift_)
    if flags[1] < 0.5:
        rate_ = np.random.uniform(1 - speed_tune, 1 + speed_tune)
        sample = librosa.effects.time_stretch(sample.astype('float32'), rate_)
    n = len(sample)
    if n < L:
        sample = np.pad(sample, (L - n, 0), 'constant', constant_values=0)
    else:
        begin = np.random.randint(0, n - L)
        sample = sample[begin:begin + L]
    if flags[2] < 0.5:
        n_noise = len(noise_samples)
        noise_ = noise_samples[np.random.randint(0, n_noise)]
        noise_begin = np.random.randint(0, len(noise_) - L)
        noise_ = noise_[noise_begin:noise_begin + L]
        volume_ = np.random.uniform(1 - volume_tune, 1 + volume_tune)
        noise_volume_ = np.random.uniform(0, noise_vol)
        sample = volume_ * sample + noise_volume_ * noise_
    return sample

