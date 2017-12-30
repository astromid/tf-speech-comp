import numpy as np
import pandas as pd
import os
import argparse
from tensorflow.python.keras.models import load_model
from utils import test_generator
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--name', dest='name', default='palsol-1')
args = parser.parse_args()

TEST_LEN = 158538
ROOT_DIR = '..'
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
SUB_DIR = os.path.join(ROOT_DIR, 'subs')
SAMPLE_SUB = os.path.join(SUB_DIR, 'sample_submission.csv')
MODEL_PATH = os.path.join(MODELS_DIR, args.name, 'model.h5')
BATCH_SIZE = 1
TOTAL = np.ceil(TEST_LEN / BATCH_SIZE).astype('int')
LABELS = 'down go left no off on right silence stop unknown up yes'.split()
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

test_gen = test_generator(batch_size=BATCH_SIZE)
model = load_model(MODEL_PATH)
pred = []
for batch in tqdm(test_gen, total=TOTAL):
    y = model.predict(batch)
    label = ID2LABEL[y]
    pred.append(label)

sub = pd.read_csv(SAMPLE_SUB)
sub['label'] = pred
sub.to_csv(os.path.join(SUB_DIR, args.name + '.csv'))
print('Submission file created successfully')
