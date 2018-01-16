import pandas as pd
import os
from tqdm import tqdm
from shutil import copyfile

ROOT_DIR = '..'
TRAIN_DIR = os.path.join(ROOT_DIR, 'data', 'train', 'audio')
TEST_DIR = os.path.join(ROOT_DIR, 'data', 'test', 'audio')
DATA = os.path.join(ROOT_DIR, 'data', 'pseudo-label.csv')
df = pd.read_csv(DATA)
os.makedirs(os.path.join(TRAIN_DIR, 'pseudo'), exist_ok=True)

for i in tqdm(range(len(df))):
    label = df.iloc[i]['label']
    if label == 'silence':
        continue
    filename = df.iloc[i]['fname']
    src = os.path.join(TEST_DIR, filename)
    if label == 'unknown':
        dst = os.path.join(TRAIN_DIR, 'pseudo', filename)
    else:
        dst = os.path.join(TRAIN_DIR, label, filename)
    copyfile(src, dst)

