import numpy as np
import time
from librosa.effects import time_stretch
from scipy.io import wavfile
from multiprocessing import Pool

dir_ = '../data/train/audio/'
BATCH_SIZE = 256
N_JOBS = 8

files = [
    'bed/989a2213_nohash_1.wav',
    'bed/9db2bfe9_nohash_0.wav',
    'bed/a1dd919f_nohash_0.wav',
    'bed/a6d586b7_nohash_0.wav',
    'bed/a7dd45cf_nohash_0.wav',
    'bed/a8cf01bc_nohash_0.wav',
    'bed/a8cf01bc_nohash_1.wav',
    'bed/a9f38bae_nohash_0.wav',
    'bed/aa753bb9_nohash_0.wav',
    'bed/ab00c4b2_nohash_0.wav',
    'bed/ab00c4b2_nohash_1.wav',
    'bed/ab7b5acd_nohash_0.wav',
    'bed/ad63d93c_nohash_0.wav',
    'bed/ae927455_nohash_0.wav',
    'bed/b00dff7e_nohash_0.wav',
    'bed/b0c0197e_nohash_0.wav']

samples = []
for file in files:
    _, sample = wavfile.read(dir_ + file)
    k = int(BATCH_SIZE / 16)
    for i in range(k):
        samples.append(sample)


def batched_time(batch_):
    for id_ in range(len(batch_)):
        rate_ = np.random.uniform(0.8, 1.2)
        batch_[id_] = time_stretch(batch_[id_].astype('float'), rate_)
    return batch_


def just_time(sample):
    rate_ = np.random.uniform(0.8, 1.2)
    sample = time_stretch(sample.astype('float'), rate_)
    return sample


def _just_speed_tune(args):
    sample, rate, flag = args
    if flag < 0.5:
        return time_stretch(sample.astype('float'), rate)
    else:
        return sample


time1 = time.time()
batch = batched_time(samples)
time2 = time.time()
delta1 = time2 - time1
print(f'Batch {BATCH_SIZE}: time = {delta1}')

batches = []
b_size = int(BATCH_SIZE / N_JOBS)
#for job in range(N_JOBS):
#    batch_crop = samples[job * b_size:(job + 1) * b_size]
#    batches.append(batch_crop)

p = Pool()
# batches = p.map(just_time, samples, chunksize=b_size)
flags = np.random.rand(BATCH_SIZE)
rates = np.random.uniform(
    1 - 0.2,
    1 + 0.2,
    BATCH_SIZE)
args = list(zip(samples, rates, flags))
batches = p.map(_just_speed_tune, args, chunksize=b_size)
time3 = time.time()
print(len(batches))
delta2 = time3 - time2
print(f'Batch {BATCH_SIZE}: time = {delta2}')
