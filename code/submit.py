import numpy as np
import pandas as pd
import os
import argparse
# from tensorflow.python.keras.models import load_model
from keras.models import load_model
from kapre.time_frequency import Melspectrogram
from utils import TestSequence2D
from scipy.stats import mode

parser = argparse.ArgumentParser()
parser.add_argument('--name', dest='name')
parser.add_argument('--best', dest='best', default=0)
parser.add_argument('--batch', dest='batch_size')
parser.add_argument('--aug', dest='augment', default=0)
parser.add_argument('--time', dest='time_shift', default=0)
parser.add_argument('--speed', dest='speed_tune', default=0)
parser.add_argument('--vol', dest='volume_tune', default=0)
parser.add_argument('--noise', dest='noise_vol', default=0)

args = parser.parse_args()

ROOT_DIR = '..'
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
SUB_DIR = os.path.join(ROOT_DIR, 'subs')

if int(args.best) == 0:
    MODEL_PATH = os.path.join(MODELS_DIR, args.name, 'model.h5')
    SUB_PATH = os.path.join(SUB_DIR, args.name + '.csv')
else:
    MODEL_PATH = os.path.join(MODELS_DIR, args.name, 'model-best.h5')
    SUB_PATH = os.path.join(SUB_DIR, args.name + '-best.csv')

BATCH_SIZE = int(args.batch_size)
N_AUG = int(args.augment)
LABELS = 'down go left no off on right silence stop unknown up yes'.split()
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
TEST_PARAMS = {
    'batch_size': BATCH_SIZE,
    'augment': 0,
    'time_shift': int(args.time_shift),
    'speed_tune': float(args.speed_tune),
    'volume_tune': float(args.volume_tune),
    'noise_vol': float(args.noise_vol),
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
ids = np.argmax(preds, axis=1)
if N_AUG != 0:
    test_seq.augment = 1
    ids_arr = [ids]
    for _ in range(N_AUG):
        preds = model.predict_generator(
            generator=test_seq,
            steps=len(test_seq),
            verbose=1
        )
        ids_arr.append(np.argmax(preds, axis=1))
    ids = mode(ids_arr)[0][0]

labels = [ID2LABEL[id_] for id_ in ids]
data = {
    'fname': test_seq.files,
    'label': labels
}
sub = pd.DataFrame(data)
sub.to_csv(SUB_PATH, index=False)
print('Submission file created successfully')
