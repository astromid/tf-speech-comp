import numpy as np
import pandas as pd
import os
from keras.models import load_model
from kapre.time_frequency import Melspectrogram
from utils import TestSequence2D


ROOT_DIR = '..'
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
SUB_DIR = os.path.join(ROOT_DIR, 'subs')
FIRST_MODEL = os.path.join(MODELS_DIR, 'seresnet-9b', 'model.h5')
MODELS = [
    os.path.join(MODELS_DIR, 'seresnet-9b', 'model-best.h5'),
    os.path.join(MODELS_DIR, 'seresnet-9a', 'model.h5'),
    os.path.join(MODELS_DIR, 'seresnet-9a', 'model-best.h5'),
    os.path.join(MODELS_DIR, 'seresnet-8a', 'model.h5'),
    os.path.join(MODELS_DIR, 'seresnet-8', 'model.h5'),
    os.path.join(MODELS_DIR, 'seresnet-7a', 'model.h5'),
]

MODELS_TTA = [
    os.path.join(MODELS_DIR, 'seresnet-9b', 'model.h5'),
    os.path.join(MODELS_DIR, 'seresnet-9b', 'model-best.h5'),
    os.path.join(MODELS_DIR, 'seresnet-9a', 'model.h5'),
    os.path.join(MODELS_DIR, 'seresnet-9a', 'model-best.h5'),
    os.path.join(MODELS_DIR, 'seresnet-8a', 'model.h5'),
]

BATCH_SIZE = 64
LABELS = 'down go left no off on right silence stop unknown up yes'.split()
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
TEST_PARAMS = {
    'batch_size': BATCH_SIZE,
    'augment': 0,
    'time_shift': 4000,
    'speed_tune': 0.3,
    'volume_tune': 0.3,
    'noise_vol': 0.3
}
test_seq = TestSequence2D(TEST_PARAMS)

model = load_model(
    FIRST_MODEL,
    custom_objects={'Melspectrogram': Melspectrogram}
)
print('Model: ' + FIRST_MODEL)
preds = model.predict_generator(
    generator=test_seq,
    steps=len(test_seq),
    verbose=1
)
preds = preds ** 0.5
for model_path in MODELS:
    print('Model: ' + model_path)
    model = load_model(
        model_path,
        custom_objects={'Melspectrogram': Melspectrogram}
    )
    curr_preds = model.predict_generator(
        generator=test_seq,
        steps=len(test_seq),
        verbose=1
    )
    preds += curr_preds ** 0.5
test_seq.augment = 1
for model_path in MODELS_TTA:
    print('Model + TTA: ' + model_path)
    model = load_model(
        model_path,
        custom_objects={'Melspectrogram': Melspectrogram}
    )
    curr_preds = model.predict_generator(
        generator=test_seq,
        steps=len(test_seq),
        verbose=1
    )
    preds += curr_preds ** 0.5
ids = np.argmax(preds, axis=1)
labels = [ID2LABEL[id_] for id_ in ids]
data = {
    'fname': test_seq.files,
    'label': labels
}
sub = pd.DataFrame(data)
sub.to_csv(os.path.join(SUB_DIR, 'ensemble-12.csv'), index=False)
print('Submission file created successfully')
