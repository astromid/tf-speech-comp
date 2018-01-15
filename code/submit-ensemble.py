import numpy as np
import pandas as pd
import os
from keras.models import load_model
from kapre.time_frequency import Melspectrogram
from utils import TestSequence2D

ROOT_DIR = '..'
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
SUB_DIR = os.path.join(ROOT_DIR, 'subs')
SUB_PATH = os.path.join(SUB_DIR, 'sub-ensemble-all-86+.csv')
NAMES = [
    'seresnet-2',
    'seresnet-4',
    'seresnet-6',
    'seresnet-7a',
    'seresnet-8',
    'seresnet-8a',

]
MODEL_PATH = os.path.join(MODELS_DIR, args.name, 'model.h5')

BATCH_SIZE = 32
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