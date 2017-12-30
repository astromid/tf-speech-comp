import numpy as np
import librosa
import os
from glob import glob
from scipy.io import wavfile

# SEED = 12017952
# np.random.seed(SEED)
ROOT_DIR = '..'
LABELS = 'down go left no off on right silence stop unknown up yes'.split()
L = 16000
TRAIN_DIR = os.path.join(ROOT_DIR, 'data', 'train', 'audio')
TEST_DIR = os.path.join(ROOT_DIR, 'data', 'test', 'audio')
VAL_LIST_PATH = os.path.join(ROOT_DIR, 'data', 'train', 'val_list.txt')
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
LABEL2ID = {label: i for i, label in ID2LABEL.items()}


def list_wav_files():
    with open(VAL_LIST_PATH) as val_txt:
        val_files = val_txt.readlines()
    all_files = [os.path.relpath(file, TRAIN_DIR) for file in
                 glob(os.path.join(TRAIN_DIR, '*', '*wav'))]
    val_files = [os.path.normpath(file)[:-1] for file in val_files]
    train_files = [file for file in all_files if file not in val_files]
    return train_files, val_files


def _pad_sample(sample):
    if len(sample) == L:
        return sample
    elif len(sample) < L:
        return np.pad(sample, (L - len(sample), 0), 'constant', constant_values=0)
    else:
        begin = np.random.randint(0, len(sample) - L)
        return sample[begin:begin + L]


def _silence_generator():
    noise_files = glob(os.path.join(TRAIN_DIR, '_background_noise_', '*wav'))
    noise_samples = []
    for file in noise_files:
        _, sample = wavfile.read(file)
        noise_samples.append(sample)
    n = len(noise_samples)
    while True:
        idx = np.random.randint(0, n)
        begin = np.random.randint(0, len(noise_samples[idx]) - L)
        yield noise_samples[idx][begin:begin + L]


def train_generator(files, batch_size, max_silence_rate):
    silence_gen = _silence_generator()
    n_train = len(files)
    while True:
        idx = 0
        # new shuffle on epoch's end
        np.random.shuffle(files)
        # 6 - number of noise files
        n_batches = np.ceil((n_train - 6) / batch_size).astype('int')
        for _ in range(n_batches):
            x_batch = []
            y_batch = []
            train_num = (1 - np.random.rand() * max_silence_rate) * batch_size
            while (len(x_batch) < train_num) and (idx < n_train):
                curr_path = files[idx]
                label = os.path.dirname(curr_path)
                f_name = os.path.basename(curr_path)
                idx += 1
                if label == '_background_noise_':
                    continue
                rate, sample = wavfile.read(os.path.join(TRAIN_DIR, label, f_name))
                sample = _pad_sample(sample)
                # augmentation should be here
                spec = librosa.feature.melspectrogram(sample, rate)
                spec = librosa.power_to_db(spec, ref=np.max)
                x_batch.append(spec)
                y = np.zeros(len(LABELS))
                if label not in LABELS:
                    label = 'unknown'
                y[LABEL2ID[label]] = 1
                y_batch.append(y)
            while len(x_batch) < batch_size:
                sample = next(silence_gen)
                spec = librosa.feature.melspectrogram(sample, L)
                spec = librosa.power_to_db(spec, ref=np.max)
                x_batch.append(spec)
                y = np.zeros(len(LABELS))
                y[LABEL2ID['silence']] = 1
                y_batch.append(y)
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
            yield x_batch.reshape(x_batch.shape + (1,)), y_batch


def val_generator(files, batch_size, silence_rate):
    silence_gen = _silence_generator()
    n_val = len(files)
    while True:
        idx = 0
        n_batches = np.ceil(n_val / batch_size).astype('int')
        for _ in range(n_batches):
            x_batch = []
            y_batch = []
            val_num = (1 - silence_rate) * batch_size
            while (len(x_batch) < val_num) and (idx < n_val):
                curr_path = files[idx]
                label = os.path.dirname(curr_path)
                f_name = os.path.basename(curr_path)
                idx += 1
                if label == '_background_noise_':
                    continue
                rate, sample = wavfile.read(os.path.join(TRAIN_DIR, label, f_name))
                sample = _pad_sample(sample)
                spec = librosa.feature.melspectrogram(sample, rate)
                spec = librosa.power_to_db(spec, ref=np.max)
                x_batch.append(spec)
                y = np.zeros(len(LABELS))
                if label not in LABELS:
                    label = 'unknown'
                y[LABEL2ID[label]] = 1
                y_batch.append(y)
            while len(x_batch) < batch_size:
                sample = next(silence_gen)
                spec = librosa.feature.melspectrogram(sample, L)
                spec = librosa.power_to_db(spec, ref=np.max)
                x_batch.append(spec)
                y = np.zeros(len(LABELS))
                y[LABEL2ID['silence']] = 1
                y_batch.append(y)
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
            yield x_batch.reshape(x_batch.shape + (1,)), y_batch


def test_generator(batch_size):
    test_files = [os.path.relpath(file, TEST_DIR) for file in
                  glob(os.path.join(TEST_DIR, '*wav'))]
    n_test = len(test_files)
    idx = 0
    n_batches = np.ceil(n_test / batch_size)
    for _ in range(n_batches):
        x_batch = []
        while (len(x_batch) < batch_size) and (idx < n_test):
            curr_path = test_files[idx]
            f_name = os.path.basename(curr_path)
            idx += 1
            rate, sample = wavfile.read(os.path.join(TEST_DIR, f_name))
            spec = librosa.feature.melspectrogram(sample, rate)
            spec = librosa.power_to_db(spec, ref=np.max)
            x_batch.append(spec)
        x_batch = np.array(x_batch)
        yield x_batch.reshape(x_batch.shape + (1,))
