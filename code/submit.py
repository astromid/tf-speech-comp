import numpy as np
import pandas as pd
import os
import argparse
from tensorflow.python.keras.models import load_model
from utils import test_generator
from utils import TestSequence
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--name', dest='name', default='palsol-1')
parser.add_argument('--batch', dest='batch_size', default=1)
args = parser.parse_args()

TEST_LEN = 158538
ROOT_DIR = '..'
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
SUB_DIR = os.path.join(ROOT_DIR, 'subs')
MODEL_PATH = os.path.join(MODELS_DIR, args.name, 'model.h5')
BATCH_SIZE = int(args.batch_size)
TOTAL = np.ceil(TEST_LEN / BATCH_SIZE).astype('int')
LABELS = 'down go left no off on right silence stop unknown up yes'.split()
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

#test_gen = test_generator(batch_size=BATCH_SIZE)
test_seq = TestSequence(BATCH_SIZE)
model = load_model(MODEL_PATH)
'''
data = {
    'fname': [],
    'label': []
}
for f_names, batch in tqdm(test_gen, total=TOTAL):
    y = model.predict(batch)
    ids = np.argmax(y, axis=1)
    labels = [ID2LABEL[id] for id in ids]
    data['fname'] += f_names
    data['label'] += labels
'''
preds = model.predict_generator(
    generator=test_seq,
    verbose=1,
    steps=len(test_seq)
)
ids = np.argmax(preds, axis=1)
labels = [ID2LABEL[id_] for id_ in ids]
data = {
    'fname': test_seq.files,
    'label': labels
}
sub = pd.DataFrame(data)
sub.to_csv(os.path.join(SUB_DIR, args.name + '.csv'), index=False)
print('Submission file created successfully')
