import numpy as np
import librosa
import os
from glob import glob
from scipy.io import wavfile
from tensorflow.python.keras.utils import Sequence

# SEED = 12017952
# np.random.seed(SEED)
LABELS = 'down go left no off on right silence stop unknown up yes'.split()
L = 16000
ROOT_DIR = '..'
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
    noise_files = glob(os.path.join(TRAIN_DIR, '_background_noise_', '*wav'))
    return train_files, val_files, noise_files


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
            # max_unknown = max_silence_rate * batch_size
            # unknown_num = 0
            while (len(x_batch) < train_num) and (idx < n_train):
                curr_path = files[idx]
                label = os.path.dirname(curr_path)
                f_name = os.path.basename(curr_path)
                idx += 1
                if label == '_background_noise_':
                    continue
                # unknown balance handling
                # if (label not in LABELS) and (unknown_num > max_unknown):
                #     continue
                rate, sample = wavfile.read(os.path.join(TRAIN_DIR, label, f_name))
                sample = _pad_sample(sample)
                # augmentation should be here
                spec = librosa.feature.melspectrogram(sample, rate)
                spec = librosa.power_to_db(spec, ref=np.max)
                x_batch.append(spec)
                y = np.zeros(len(LABELS))
                if label not in LABELS:
                    label = 'unknown'
                    # unknown_num += 1
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
    n_batches = np.ceil(n_test / batch_size).astype('int')
    for _ in range(n_batches):
        x_batch = []
        f_names = []
        while (len(x_batch) < batch_size) and (idx < n_test):
            curr_path = test_files[idx]
            f_name = os.path.basename(curr_path)
            f_names.append(f_name)
            idx += 1
            rate, sample = wavfile.read(os.path.join(TEST_DIR, f_name))
            spec = librosa.feature.melspectrogram(sample, rate)
            spec = librosa.power_to_db(spec, ref=np.max)
            x_batch.append(spec)
        x_batch = np.array(x_batch)
        yield f_names, x_batch.reshape(x_batch.shape + (1,))


class TrainSequence(Sequence):

    def __init__(self, files, noise_files, params):
        self.files = files
        self.noise_files = noise_files
        self.full_batch_size = params['batch_size']
        self.augment = params['augment']
        self.silence_rate = params['silence_rate']
        self.time_shift = params['time_shift']
        self.speed_tune = params['speed_tune']
        self.volume_tune = params['volume_tune']
        self.noise_vol = params['noise_vol']
        self.batch_size = int((1 - self.silence_rate) * self.full_batch_size)

    def __len__(self):
        return np.ceil(len(self.files) / self.batch_size).astype('int')

    @property
    def __get_silence(self):
        n = len(self.noise_files)
        file = self.noise_files[np.random.randint(0, n)]
        _, sample = wavfile.read(file)
        return self.__pad_sample(sample)

    def __time_shift(self, sample):
        shift_ = int(np.random.uniform(-self.time_shift, self.time_shift))
        return np.roll(sample, shift_)

    def __speed_tune(self, sample):
        rate_ = np.random.uniform(1-self.speed_tune, 1+self.speed_tune)
        return librosa.effects.time_stretch(sample.astype('float32'), rate_)

    def __get_noised(self, sample):
        noise_ = self.__get_silence
        volume_ = np.random.uniform(1-self.volume_tune, 1+self.volume_tune)
        noise_volume_ = np.random.uniform(0, self.noise_vol)
        return volume_ * sample + noise_volume_ * noise_

    def __pad_sample(self, sample):
        n = len(sample)
        if n == L:
            return sample
        elif n < L:
            return np.pad(sample, (L - n, 0), 'constant', constant_values=0)
        else:
            begin = np.random.randint(0, n - L)
            return sample[begin:begin + L]

    def __get_sample(self, file):
        label = os.path.dirname(file)
        f_name = os.path.basename(file)
        rate, sample = wavfile.read(os.path.join(TRAIN_DIR, label, f_name))
        # augmentation should be here
        if self.augment is 'yes':
            sample = self.__time_shift(sample)
            sample = self.__speed_tune(sample)
            sample = self.__pad_sample(sample)
            if np.random.rand() < 0.5:
                sample = self.__get_noised(sample)
            sample = sample.astype('int16')
        else:
            sample = self.__pad_sample(sample)
        spect = librosa.feature.melspectrogram(sample, rate)
        spect = librosa.power_to_db(spect, ref=np.max)
        y = np.zeros(len(LABELS))
        if label not in LABELS:
            label = 'unknown'
        y[LABEL2ID[label]] = 1
        return spect, y

    def __getitem__(self, idx):
        file_batch = self.files[idx * self.batch_size:(idx + 1) * self.batch_size]
        x_batch = []
        y_batch = []
        for file in file_batch:
            x, y = self.__get_sample(file)
            x_batch.append(x)
            y_batch.append(y)
        while len(x_batch) < self.full_batch_size:
            x = self.__get_silence
            spect = librosa.feature.melspectrogram(x, L)
            spect = librosa.power_to_db(spect, ref=np.max)
            x_batch.append(spect)
            y = np.zeros(len(LABELS))
            y[LABEL2ID['silence']] = 1
            y_batch.append(y)
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        x_batch = x_batch.reshape(x_batch.shape + (1,))
        return x_batch, y_batch

    def on_epoch_end(self):
        np.random.shuffle(self.files)


class ValSequence(Sequence):

    def __init__(self, files, noise_files, params):
        self.files = files
        self.noise_files = noise_files
        self.full_batch_size = params['batch_size']
        self.silence_rate = params['silence_rate']
        self.batch_size = int((1 - self.silence_rate) * self.full_batch_size)

    def __len__(self):
        return np.ceil(len(self.files) / self.batch_size).astype('int')

    @property
    def __get_silence(self):
        n = len(self.noise_files)
        file = self.noise_files[np.random.randint(0, n)]
        _, sample = wavfile.read(file)
        return self.__pad_sample(sample)

    def __pad_sample(self, sample):
        n = len(sample)
        if n == L:
            return sample
        elif n < L:
            return np.pad(sample, (L - n, 0), 'constant', constant_values=0)
        else:
            begin = np.random.randint(0, n - L)
            return sample[begin:begin + L]

    def __get_sample(self, file):
        label = os.path.dirname(file)
        f_name = os.path.basename(file)
        rate, sample = wavfile.read(os.path.join(TRAIN_DIR, label, f_name))
        sample = self.__pad_sample(sample)
        spect = librosa.feature.melspectrogram(sample, rate)
        spect = librosa.power_to_db(spect, ref=np.max)
        y = np.zeros(len(LABELS))
        if label not in LABELS:
            label = 'unknown'
        y[LABEL2ID[label]] = 1
        return spect, y

    def __getitem__(self, idx):
        file_batch = self.files[idx * self.batch_size:(idx + 1) * self.batch_size]
        x_batch = []
        y_batch = []
        for file in file_batch:
            x, y = self.__get_sample(file)
            x_batch.append(x)
            y_batch.append(y)
        while len(x_batch) < self.full_batch_size:
            x = self.__get_silence
            spect = librosa.feature.melspectrogram(x, L)
            spect = librosa.power_to_db(spect, ref=np.max)
            x_batch.append(spect)
            y = np.zeros(len(LABELS))
            y[LABEL2ID['silence']] = 1
            y_batch.append(y)
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        x_batch = x_batch.reshape(x_batch.shape + (1,))
        return x_batch, y_batch

    def on_epoch_end(self):
        pass


class TestSequence(Sequence):

    def __init__(self, params):
        self.files = [os.path.relpath(file, TEST_DIR) for file in
                      glob(os.path.join(TEST_DIR, '*wav'))]
        self.batch_size = params['batch_size']

    def __len__(self):
        return np.ceil(len(self.files) / self.batch_size).astype('int')

    @staticmethod
    def __get_sample(file):
        f_name = os.path.basename(file)
        rate, sample = wavfile.read(os.path.join(TEST_DIR, f_name))
        spect = librosa.feature.melspectrogram(sample, rate)
        spect = librosa.power_to_db(spect, ref=np.max)
        return spect

    def __getitem__(self, idx):
        file_batch = self.files[idx * self.batch_size:(idx + 1) * self.batch_size]
        x_batch = []
        # f_names = []
        for file in file_batch:
            x_batch.append(self.__get_sample(file))
        x_batch = np.array(x_batch)
        x_batch = x_batch.reshape(x_batch.shape + (1,))
        return x_batch

    def on_epoch_end(self):
        pass
