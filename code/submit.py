import numpy as np
import pandas as pd
import os
import argparse
from keras.models import load_model
from kapre.time_frequency import Melspectrogram
from utils import TestSequence2D

parser = argparse.ArgumentParser()
parser.add_argument('--name')
parser.add_argument('--best', type=int, default=0)
parser.add_argument('--batch', type=int)
parser.add_argument('--aug', type=int, default=0)
parser.add_argument('--time', type=int, default=0)
parser.add_argument('--speed', type=float, default=0.0)
parser.add_argument('--vol', type=float, default=0.0)
parser.add_argument('--noise', type=float, default=0.0)

args = parser.parse_args()

ROOT_DIR = '..'
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
SUB_DIR = os.path.join(ROOT_DIR, 'subs')
N_AUG = args.aug

if N_AUG == 0:
    sub_end = '.csv'
else:
    sub_end = f'-tta-{N_AUG}.csv'

if args.best == 0:
    MODEL_PATH = os.path.join(MODELS_DIR, args.name, 'model.h5')
    SUB_PATH = os.path.join(SUB_DIR, args.name + sub_end)
else:
    MODEL_PATH = os.path.join(MODELS_DIR, args.name, 'model-best.h5')
    SUB_PATH = os.path.join(SUB_DIR, args.name + '-best' + sub_end)

BATCH_SIZE = args.batch
LABELS = 'down go left no off on right silence stop unknown up yes'.split()
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
TEST_PARAMS = {
    'batch_size': BATCH_SIZE,
    'augment': 0,
    'time_shift': args.time,
    'speed_tune': args.speed,
    'volume_tune': args.vol,
    'noise_vol': args.noise,
}
model = load_model(
    MODEL_PATH,
    custom_objects={'Melspectrogram': Melspectrogram}
)

test_seq = TestSequence2D(TEST_PARAMS)
preds = model.predict_generator(
    generator=test_seq,
    steps=len(test_seq),
    verbose=1
)
if N_AUG != 0:
    preds = preds ** 0.4
    test_seq.augment = 1
    for _ in range(N_AUG):
        aug_preds = model.predict_generator(
            generator=test_seq,
            steps=len(test_seq),
            verbose=1
        )
        preds += aug_preds ** 0.4
ids = np.argmax(preds, axis=1)
labels = [ID2LABEL[id_] for id_ in ids]
data = {
    'fname': test_seq.files,
    'label': labels
}
sub = pd.DataFrame(data)
sub.to_csv(SUB_PATH, index=False)
print('Submission file created successfully')
