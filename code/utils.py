import numpy as np
import os
import inspect
from glob import glob
from scipy.io import wavfile
from keras.utils import Sequence
from librosa.effects import time_stretch
from keras.callbacks import Callback
from tqdm import tqdm
from sklearn.utils.class_weight import compute_sample_weight
from multiprocessing import Pool

# SEED = 12017952
# np.random.seed(SEED)
LABELS = 'down go left no off on right silence stop unknown up yes'.split()
L = 16000
N_CLASS = len(LABELS)
ROOT_DIR = '..'
TRAIN_DIR = os.path.join(ROOT_DIR, 'data', 'train', 'audio')
TEST_DIR = os.path.join(ROOT_DIR, 'data', 'test', 'audio')
VAL_LIST_PATH = os.path.join(ROOT_DIR, 'data', 'train', 'val_list.txt')
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
LABEL2ID = {label: i for i, label in ID2LABEL.items()}
N_JOBS = os.cpu_count()

# change built-in print with tqdm.write
old_print = print


def tqdm_print(*args, **kwargs):
    try:
        tqdm.write(*args, **kwargs)
    except:
        old_print(*args, **kwargs)


inspect.builtins.print = tqdm_print


def _batched_speed_tune(batch, speed_tune):
    n = len(batch)
    minibatch_size = np.ceil(n / N_JOBS).astype('int')
    flags = np.random.rand(n)
    rates = np.random.uniform(1 - speed_tune, 1 + speed_tune, n)
    args = list(zip(batch, rates, flags))
    p = Pool()
    batch = p.map(_just_speed_tune, args, chunksize=minibatch_size)
    return batch


def _just_speed_tune(args):
    sample, rate, flag = args
    if flag < 0.5:
        return time_stretch(sample.astype('float'), rate)
    else:
        return sample


class AudioSequence(Sequence):

    def __init__(self, params):
        # properties to determine in Train & Val seqs
        self.files = None
        self.batch_size = None
        self.known = None
        self.unknown = None
        self.labels = None
        self.n_silence = None
        self.n_unknown = None
        self.eps = None
        self.balance = None

        # shared properties
        self.noise_samples = self._load_noise_samples
        self.full_batch_size = params['batch_size']
        self.augment = params['augment']
        self.time_shift = params['time_shift']
        self.speed_tune = params['speed_tune']
        self.volume_tune = params['volume_tune']
        self.noise_vol = params['noise_vol']

    def __len__(self):
        return np.ceil(len(self.known) / self.batch_size).astype('int')

    def __getitem__(self, idx):
        x = self.known[idx * self.batch_size:(idx + 1) * self.batch_size]
        y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        for _ in range(self.n_silence):
            x.append(self._get_silence)
            y.append('silence')
        unknown_idx = np.random.randint(0, len(self.unknown), self.n_unknown)
        for idx in unknown_idx:
            x.append(self.unknown[idx])
            y.append('unknown')
        label_ids = [LABEL2ID[label] for label in y]
        if self.augment == 0:
            batch = [self._pad_sample(s) for s in x]
        else:
            # batch = [self._augment_sample(s) for s in x]
            batch = self._augment_batch(x)
        ohe_batch = []
        for id_ in label_ids:
            ohe_y = np.ones(N_CLASS) * self.eps / (N_CLASS - 1)
            ohe_y[id_] = 1 - self.eps
            ohe_batch.append(ohe_y)
        batch = np.array(batch)
        ohe_batch = np.array(ohe_batch)
        batch = batch.reshape((-1, 1, L))
        if self.balance == 0:
            return batch, ohe_batch
        else:
            weights = compute_sample_weight('balanced', label_ids)
            return batch, ohe_batch, weights

    @property
    def _list_val_files(self):
        with open(VAL_LIST_PATH) as val_txt:
            val_files = val_txt.readlines()
        val_files = [os.path.normpath(file)[:-1] for file in val_files]
        return val_files

    @property
    def _list_train_files(self):
        all_files = [os.path.relpath(file, TRAIN_DIR) for file in
                     glob(os.path.join(TRAIN_DIR, '*', '*wav'))]
        val_files = self._list_val_files
        train_files = [file for file in all_files if file not in val_files]
        return train_files

    @property
    def _list_test_files(self):
        test_files = [os.path.relpath(file, TEST_DIR) for file in
                      glob(os.path.join(TEST_DIR, '*wav'))]
        return test_files

    @property
    def _load_noise_samples(self):
        noise_files = glob(os.path.join(TRAIN_DIR, '_background_noise_', '*wav'))
        noise_samples = []
        for file in noise_files:
            _, sample = wavfile.read(file)
            noise_samples.append(sample)
        noise_samples.append(np.zeros(L))
        return noise_samples

    @property
    def _load_samples(self):
        known = []
        unknown = []
        labels = []
        for file in tqdm(self.files, desc='Loading files'):
            label = os.path.dirname(file)
            if label is '_background_noise_':
                continue
            f_name = os.path.basename(file)
            rate, sample = wavfile.read(os.path.join(TRAIN_DIR, label, f_name))
            if label not in LABELS:
                unknown.append(sample)
            else:
                known.append(sample)
                labels.append(label)
        assert len(known) == len(labels)
        return known, unknown, labels

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

    def _time_shift(self, sample):
        shift_ = int(np.random.uniform(-self.time_shift, self.time_shift))
        return np.roll(sample, shift_)

    def _speed_tune(self, sample):
        rate_ = np.random.uniform(1 - self.speed_tune, 1 + self.speed_tune)
        return time_stretch(sample.astype('float'), rate_)

    def _get_noised(self, sample):
        noise_ = self._get_silence
        volume_ = np.random.uniform(1 - self.volume_tune, 1 + self.volume_tune)
        noise_volume_ = np.random.uniform(0, self.noise_vol)
        return volume_ * sample + noise_volume_ * noise_

    def _augment_sample(self, sample):
        flags = np.random.rand(3)
        if self.time_shift != 0 and flags[0] < 0.5:
            sample = self._time_shift(sample)
        if self.speed_tune != 0 and flags[1] < 0.5:
            sample = self._speed_tune(sample)
        sample = self._pad_sample(sample)
        if self.noise_vol != 0 and flags[2] < 0.5:
            sample = self._get_noised(sample)
        return sample

    def _augment_batch(self, batch):
        n = len(batch)
        if self.time_shift != 0:
            for i in range(n):
                if np.random.rand() < 0.5:
                    batch[i] = self._time_shift(batch[i])
        if self.speed_tune != 0:
            # minibatch_size = np.ceil(n / N_JOBS).astype('int')
            '''
            minibatches = []
            for i in range(N_JOBS):
                minibatch = batch[i * minibatch_size:(i + 1) * minibatch_size]
                minibatches.append(minibatch)
            # p = Pool()
            results = self.p.map(_batched_speed_tune, minibatches)
            batch = [item for sublist in results for item in sublist]
            '''
            batch = _batched_speed_tune(batch, self.speed_tune)
        if self.noise_vol != 0:
            for i in range(n):
                if np.random.rand() < 0.5:
                    batch[i] = self._get_noised(batch[i])
        return batch

    def on_epoch_end(self):
        pass


class TrainSequence2D(AudioSequence):

    def __init__(self, params):
        super().__init__(params)
        self.files = self._list_train_files
        self.known, self.unknown, self.labels = self._load_samples
        # shuffle before start
        self.on_epoch_end()
        self.n_silence = np.ceil(params['silence'] * self.full_batch_size).astype('int')
        self.n_unknown = np.ceil(params['unknown'] * self.full_batch_size).astype('int')
        self.eps = params['eps']
        self.balance = params['balance']
        self.batch_size = self.full_batch_size - self.n_unknown - self.n_silence

    def on_epoch_end(self):
        data = list(zip(self.known, self.labels))
        np.random.shuffle(data)
        self.known, self.labels = zip(*data)
        self.known = list(self.known)
        self.labels = list(self.labels)


class ValSequence2D(AudioSequence):

    def __init__(self, params):
        super().__init__(params)
        self.files = self._list_val_files
        self.known, self.unknown, self.labels = self._load_samples
        self.n_silence = np.ceil(params['silence'] * self.full_batch_size).astype('int')
        # validation uses all unknowns
        # self.n_unknown = np.ceil(params['unknown'] * self.full_batch_size).astype('int')
        self.n_unknown = 0
        self.known += self.unknown
        self.labels += ['unknown'] * len(self.unknown)
        self.eps = params['eps']
        # no balance at validation time
        self.balance = 0
        # self.batch_size = self.full_batch_size - self.n_unknown - self.n_silence
        self.batch_size = self.full_batch_size - self.n_silence


class TestSequence2D(AudioSequence):

    def __init__(self, params):
        super().__init__(params)
        self.files = self._list_test_files
        self.known = self._load_samples
        self.batch_size = self.full_batch_size

    def __getitem__(self, idx):
        x = self.known[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.augment == 0:
            batch = [self._pad_sample(s) for s in x]
        else:
            batch = [self._augment_sample(s) for s in x]
        batch = np.array(batch)
        batch = batch.reshape((-1, 1, L))
        return batch

    @property
    def _load_samples(self):
        samples = []
        for file in tqdm(self.files, desc='Loading files'):
            f_name = os.path.basename(file)
            rate, sample = wavfile.read(os.path.join(TEST_DIR, f_name))
            samples.append(sample)
        return samples


class LoggerCallback(Callback):

    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs={}):
        metrics = self.params['metrics']
        metric_format = '{name}: {value:0.3f}'
        strings = [metric_format.format(
            name=metric,
            value=np.mean(logs[metric], axis=None)
        ) for metric in metrics if metric in logs]
        epoch_output = 'Epoch {value:05d}: '.format(value=(epoch + 1))
        output = epoch_output + ', '.join(strings)
        print(output)
