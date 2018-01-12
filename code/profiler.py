from utils import ValSequence2D

TRAIN_PARAMS = {
    'batch_size': 64,
    'balance': 1,
    'eps': 0,
    'silence': 0.1,
    'unknown': 0.1,
    'augment': 1,
    'time_shift': 2000,
    'speed_tune': 0.2,
    'volume_tune': 0.2,
    'noise_vol': 0.4
}

val_seq = ValSequence2D(TRAIN_PARAMS)
for idx in range(100):
    A = val_seq.__getitem__(idx)
