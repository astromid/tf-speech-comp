import numpy as np
import os
import argparse
import models
from utils import list_wav_files, train_generator, val_generator
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument('--batch', dest='batch_size', default=64)
parser.add_argument('--silence', dest='max_silence_rate', default=0.2)
args = parser.parse_args()

ROOT_DIR = '..'
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
BATCH_SIZE = args.batch_size
MAX_SILENCE_RATE = args.max_silence_rate

train_files, val_files = list_wav_files()
train_gen = train_generator(train_files, BATCH_SIZE, MAX_SILENCE_RATE)
val_gen = val_generator(val_files, BATCH_SIZE, MAX_SILENCE_RATE)
model = models.palsol_model()
